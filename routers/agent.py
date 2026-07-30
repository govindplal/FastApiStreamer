import json
import re
import asyncio

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
async def run_agent(request: AgentRequest, db: AsyncSession = Depends(get_db)):

    db_session = Session(task=request.prompt)
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)

    # Log the initial user message
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
        
        # State tracker to prevent infinite loops
        past_actions = {}

        def sse_event(event_type: str, content: any):
            payload = json.dumps({"type": event_type, "content": content})
            return f"data: {payload}\n\n"

        async def safe_dispatch(name: str, args: dict) -> str:
            action_signature = f"{name}:{json.dumps(args, sort_keys=True)}"
            attempts = past_actions.get(action_signature, 0)
            
            # Allow a maximum of 2 identical attempts before blacklisting
            if attempts >= 2:
                logger.warning(f"Prevented duplicate tool call loop: {name}")
                return "System Error: You have repeatedly executed this exact tool with these exact arguments and failed. Do not repeat this action. Try a different approach or provide your final answer."
            
            past_actions[action_signature] = attempts + 1

            try:
                # 30-second hard limit per tool
                result = await asyncio.wait_for(dispatch_tool(name, args), timeout=30.0)
                return str(result)
            except asyncio.TimeoutError:
                error_msg = f"System Error: Tool '{name}' timed out after 30 seconds. You may retry."
                logger.error(error_msg)
                return error_msg
            except Exception as e:
                error_msg = f"System Error: Tool '{name}' failed with error: {str(e)}. You may retry."
                logger.error(error_msg)
                return error_msg


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

            accumulated_content = ""
            accumulated_tool_calls = {}

            async for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Accumulate text and stream it instantly
                content = delta.content or delta.refusal
                if content:
                    accumulated_content += content
                    yield sse_event("text_delta", content)

                # Accumulate tool call fragments but do not stream them
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
                            if tc_chunk.function and tc_chunk.function.name:
                                accumulated_tool_calls[idx]["name"] += tc_chunk.function.name
                            if tc_chunk.function and tc_chunk.function.arguments:
                                accumulated_tool_calls[idx]["arguments"] += tc_chunk.function.arguments

            # Format tool calls for the message history
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

            # Native Tool Calls
            if accumulated_tool_calls:
                for idx, tool_data in accumulated_tool_calls.items():
                    tool_name = tool_data["name"]
                    try:
                        parsed_arguments = json.loads(tool_data["arguments"])
                    except json.JSONDecodeError:
                        parsed_arguments = {}

                    logger.info(f"Acting: Executing {tool_name}")
                    yield sse_event("tool_call", {"name": tool_name, "arguments": parsed_arguments})

                    tool_output = await safe_dispatch(tool_name, parsed_arguments)

                    db_tool = ToolCall(session_id=db_session.id, tool_name=tool_name, tool_input=parsed_arguments, tool_result={"output": tool_output})
                    db.add(db_tool)
                    db_tool_msg = Message(session_id=db_session.id, role="tool", content=tool_output)
                    db.add(db_tool_msg)
                    await db.commit()

                    yield sse_event("tool_result", tool_output)

                    messages.append({"role": "tool", "tool_call_id": tool_data["id"], "name": tool_name, "content": tool_output})
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
                            
                            tool_output = await safe_dispatch(tool_name, parsed_arguments)
                            
                            db_tool = ToolCall(session_id=db_session.id, tool_name=tool_name, tool_input=parsed_arguments, tool_result={"output": tool_output})
                            db.add(db_tool)
                            db_tool_msg = Message(session_id=db_session.id, role="tool", content=f"Tool {tool_name} returned: {tool_output}")
                            db.add(db_tool_msg)
                            await db.commit()
                            
                            yield sse_event("tool_result", tool_output)
                            
                            messages.append({
                                "role": "user", 
                                "content": f"The tool '{tool_name}' returned: {tool_output}. Continue your task."
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