import json
import re

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from tools.registry import dispatch_tool
from tools.schemas import AGENT_TOOLS

from core.config import settings
from core.llm import client

from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from models.db import Session, Message, ToolCall

SYSTEM_PROMPT = """You are Aperture, an autonomous AI agent.
You operate using a strict Think -> Act -> Observe loop.

1. First, THINK through what you need to do based on the user's request. Keep your thinking concise.
2. If you need external data or capabilities, ACT by calling a tool. 
3. You will receive an OBSERVATION (the tool's result).
4. Repeat this Think -> Act -> Observe cycle until you have enough information to fulfill the request.

When the task is complete, do not call any more tools. Provide your final response directly to the user explicitly.
"""

router = APIRouter(prefix="/agent", tags=["Agent Operations"])

class AgentRequest(BaseModel):
    prompt: str

@router.post("/run")
async def run_agent(request: AgentRequest, db:AsyncSession = Depends(get_db)):

    db_session = Session(task=request.prompt)
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session) # This pulls the auto-generated ID back from Postgres

    # 2. Log the initial user message
    db_msg = Message(
        session_id=db_session.id,
        role="user",
        content=request.prompt
    )
    db.add(db_msg)
    await db.commit()

    logger.info(f"Starting ReAct loop for task: {request.prompt}")

    async def agent_stream_generator():

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt}
        ]

        # Format data for Server Sent Events (SSE)
        def sse_event(event_type: str, content: any):
            payload = json.dumps({"type": event_type, "content": content})
            return f"data: {payload}\n\n"

        max_iterations = 10
        for iteration in range(max_iterations):
            logger.info(f"--- Loop Iteration {iteration + 1} ---")

            response = await client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                stream=True
            )

            #Accumulators to trap tool chunks to make streaming and tool executions possible
            accumulated_content = ""
            accumulated_tool_calls = {}

            async for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                #Accumulate text and stream it instantly
                if delta.content:
                    accumulated_content += delta.content
                    yield sse_event("text_delta", delta.content)

                #Accumulate tool call fragments but do not stream them
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc_chunk.id,
                                "name": tc_chunk.function.name or "",
                                "arguments": tc_chunk.function.arguments or ""
                            }
                        else:
                            if tc_chunk.function.name:
                                accumulated_tool_calls[idx]["name"] += tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += tc_chunk.function.arguments

            assistant_message = {"role": "assistant", "content": accumulated_content}

            if accumulated_tool_calls:
                formatted_tool_calls = []
                for idx, tc in accumulated_tool_calls.items():
                    formatted_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    })
                assistant_message["tool_calls"] = formatted_tool_calls

            messages.append(assistant_message)

            if accumulated_tool_calls:

                for idx, tool_data in accumulated_tool_calls.items():
                    tool_name = tool_data["name"]
                    try:
                        parsed_arguments = json.loads(tool_data["arguments"])
                    except json.JSONDecodeError:
                        parsed_arguments = {}

                    logger.info(f"Acting: Executing {tool_name}")
                    yield sse_event("tool_call", {"name": tool_name, "arguments": parsed_arguments})

                    tool_output = await dispatch_tool(tool_name, parsed_arguments)

                    db_tool = ToolCall(session_id=db_session.id, tool_name=tool_name, tool_input=parsed_arguments, tool_result={"output": tool_output})
                    db.add(db_tool)
                    db_tool_msg = Message(session_id=db_session.id, role="tool", content=str(tool_output))
                    db.add(db_tool_msg)
                    await db.commit()

                    yield sse_event("tool_result", tool_output)

                    messages.append({"role": "tool", "tool_call_id": tool_data["id"], "name": tool_name, "content": str(tool_output)})
                continue

            # Rogue JSON Trap
            elif accumulated_content:
                try:
                    json_match = re.search(r'\{.*\}', accumulated_content, re.DOTALL)
                    if json_match:
                        possible_tool_call = json.loads(json_match.group(0))
                        
                        if isinstance(possible_tool_call, dict) and "name" in possible_tool_call and "arguments" in possible_tool_call:
                            tool_name = possible_tool_call["name"]
                            parsed_arguments = possible_tool_call["arguments"]
                            
                            logger.info(f"Caught rogue JSON tool call: {tool_name}")
                            yield sse_event("tool_call", {"name": tool_name, "arguments": parsed_arguments})
                            
                            tool_result_str = await dispatch_tool(tool_name, parsed_arguments)
                            
                            # Log to DB
                            db_tool = ToolCall(session_id=db_session.id, tool_name=tool_name, tool_input=parsed_arguments, tool_result={"output": tool_result_str})
                            db.add(db_tool)
                            db_tool_msg = Message(session_id=db_session.id, role="tool", content=f"Tool {tool_name} returned: {tool_result_str}")
                            db.add(db_tool_msg)
                            await db.commit()
                            
                            yield sse_event("tool_result", tool_result_str)
                            
                            messages.append({
                                "role": "user", 
                                "content": f"The tool '{tool_name}' returned: {tool_result_str}. Continue your task."
                            })
                            continue
                except json.JSONDecodeError:
                    pass
                
                db_msg = Message(session_id=db_session.id, role="assistant", content=accumulated_content)
                db.add(db_msg)
                
                yield sse_event("done", "")
                db_session.status = "complete"
                await db.commit()
                return

        yield sse_event("error", "Agent reached maximum iterations.")
        db_session.status = "error"
        await db.commit()

    return StreamingResponse(agent_stream_generator(), media_type="text/event-stream")
