import json
import re
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from core.graph import GraphTracker, ReplayOverrideRequest
from tools.registry import dispatch_tool
from tools.schemas import AGENT_TOOLS

from core.config import settings
from core.llm import client
from core.memory import store_memory

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
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
async def run_agent(request: Request, agent_request: AgentRequest, db: AsyncSession = Depends(get_db)):

    db_session = Session(task=agent_request.prompt)
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)

    db_msg = Message(
        session_id=db_session.id,
        role="user",
        content=agent_request.prompt
    )
    db.add(db_msg)
    await db.commit()

    logger.info(f"Starting ReAct loop for task: {agent_request.prompt}")

    graph = GraphTracker()
    graph.add_node("user_prompt",agent_request.prompt)
    async def agent_stream_generator():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": agent_request.prompt}
        ]
        
        past_actions = {}
        exit_reason = "max_iterations"  

        def sse_event(event_type: str, content: any):
            payload = json.dumps({"type": event_type, "content": content})
            return f"data: {payload}\n\n"

        async def safe_dispatch(name: str, args: dict) -> str:
            action_signature = f"{name}:{json.dumps(args, sort_keys=True)}"
            attempts = past_actions.get(action_signature, 0)
            
            if attempts >= 2:
                logger.warning(f"Prevented duplicate tool call loop: {name}")
                return "System Error: You have repeatedly executed this exact tool with these exact arguments and failed. Do not repeat this action. Try a different approach or provide your final answer."
            
            past_actions[action_signature] = attempts + 1

            try:
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
            # Check client connectivity before requesting the LLM turn
            if await request.is_disconnected():
                logger.warning("Client disconnected. Halting ReAct loop.")
                exit_reason = "disconnected"
                break

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

                content = delta.content or delta.refusal
                if content:
                    accumulated_content += content
                    yield sse_event("text_delta", content)

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

            assistant_message = {"role": "assistant", "content": accumulated_content}

            if accumulated_content and accumulated_tool_calls:
                graph.add_node("think", accumulated_content)
        
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

            if not accumulated_content and not accumulated_tool_calls:
                logger.error("LLM returned an empty response.")
                exit_reason = "empty_response"
                break

            # Path A: Concurrent Native Tool Execution
            if accumulated_tool_calls:
                dispatch_tasks = []
                tool_metadata = []

                for idx, tool_data in accumulated_tool_calls.items():
                    tool_name = tool_data["name"]
                    try:
                        parsed_arguments = json.loads(tool_data["arguments"])
                    except json.JSONDecodeError:
                        parsed_arguments = {}

                    logger.info(f"Acting: Queuing {tool_name}")
                    yield sse_event("tool_call", {"name": tool_name, "arguments": parsed_arguments})

                    graph.add_node("tool_call", {"name": tool_name, "arguments": parsed_arguments})
                    
                    dispatch_tasks.append(safe_dispatch(tool_name, parsed_arguments))
                    tool_metadata.append((tool_data["id"], tool_name, parsed_arguments))

                # Concurrently execute tools 
                tool_outputs = await asyncio.gather(*dispatch_tasks)

                for (tool_call_id, tool_name, parsed_arguments), tool_output in zip(tool_metadata, tool_outputs):

                    graph.add_node("tool_result", {"name": tool_name, "output": tool_output})

                    if tool_output and not tool_output.startswith("System Error:"):
                        await store_memory(content=f"Tool '{tool_name}' observed: {tool_output[:1000]}", db=db)
                    
                    db_tool = ToolCall(session_id=db_session.id, tool_name=tool_name, tool_input=parsed_arguments, tool_result={"output": tool_output})
                    db.add(db_tool)
                    db_tool_msg = Message(session_id=db_session.id, role="tool", content=tool_output)
                    db.add(db_tool_msg)
                    await db.commit()

                    yield sse_event("tool_result", tool_output)
                    messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": tool_output})
                continue

            # Path B: Rogue JSON Trap
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
                            
                            graph.add_node("tool_call", {"name": tool_name, "arguments": parsed_arguments})

                            tool_output = await safe_dispatch(tool_name, parsed_arguments)

                            if tool_output and not tool_output.startswith("System Error:"):
                                await store_memory(content=f"Tool '{tool_name}' observed: {tool_output[:1000]}", db=db)
                            
                            db_tool = ToolCall(session_id=db_session.id, tool_name=tool_name, tool_input=parsed_arguments, tool_result={"output": tool_output})
                            db.add(db_tool)
                            db_tool_msg = Message(session_id=db_session.id, role="tool", content=f"Tool {tool_name} returned: {tool_output}")
                            db.add(db_tool_msg)
                            await db.commit()
                            
                            yield sse_event("tool_result", tool_output)

                            graph.add_node("tool_result", {"name": tool_name, "output": tool_output})
                            
                            messages.append({
                                "role": "user", 
                                "content": f"[Tool Observation for '{tool_name}']: {tool_output}. Continue your task."
                            })
                            continue
                except json.JSONDecodeError:
                    pass
                
                db_msg = Message(session_id=db_session.id, role="assistant", content=accumulated_content)
                db.add(db_msg)

                graph.add_node("done", accumulated_content)
                
                db_session.workflow_graph = graph.nodes
                db_session.status = "complete"

                await db.commit()

                yield sse_event("done", "")
                return

        if exit_reason == "disconnected":
            db_session.status = "cancelled"
            db_session.workflow_graph = graph.nodes
            await db.commit()
        elif exit_reason == "empty_response":
            yield sse_event("error", "Agent received empty response from LLM provider.")
            db_session.status = "error"
            db_session.workflow_graph = graph.nodes
            await db.commit()
        elif exit_reason == "max_iterations":
            yield sse_event("error", "Agent reached maximum iterations.")
            db_session.status = "error"
            db_session.workflow_graph = graph.nodes
            await db.commit()

    return StreamingResponse(agent_stream_generator(), media_type="text/event-stream")

@router.get("/{session_id}/replay")
async def replay_session(session_id: int, db: AsyncSession = Depends(get_db)):
    query = select(Session).where(Session.id == session_id)
    result = await db.execute(query)
    db_session = result.scalar_one_or_none()

    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def replay_generator():
        if not db_session.workflow_graph:
            error_payload = json.dumps({"type": "error", "content": "No workflow graph found for this session."})
            yield f"data: {error_payload}\n\n"
            return

        for node in db_session.workflow_graph:

            payload = json.dumps({
                "id": node["id"],
                "type": node["type"],
                "content": node["content"],
            })

            yield f"data: {payload}\n\n"

            await asyncio.sleep(0.5)
        
    return StreamingResponse(replay_generator(), media_type="text/event-stream")

@router.post("/{session_id}/replay")
async def override_replay_session(
    session_id: int, 
    override_req: ReplayOverrideRequest, 
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    query = select(Session).where(Session.id == session_id)
    result = await db.execute(query)
    old_session = result.scalar_one_or_none()

    if not old_session or not old_session.workflow_graph:
        raise HTTPException(status_code=404, detail="Session or graph not found")

    #Create a new DB Session for this alternate timeline
    new_db_session = Session(task=f"Replay of Session {session_id} (Node Override)")
    db.add(new_db_session)
    await db.commit()
    await db.refresh(new_db_session)

    logger.info(f"Starting Override Replay for Session {session_id}")

    async def replay_override_generator():
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        graph = GraphTracker()
        
        def sse_event(event_type: str, content: any):
            payload = json.dumps({"type": event_type, "content": content})
            return f"data: {payload}\n\n"

        #Rebuild history up to the target node
        target_found = False

        active_tool_ids = {}
        for node in old_session.workflow_graph:
            node_type = node["type"]
            content = node["content"]
            
            if node_type == "user_prompt":
                messages.append({"role": "user", "content": content})
                graph.add_node("user_prompt", content)
            
            elif node_type == "think":
                messages.append({"role": "assistant", "content": content})
                graph.add_node("think", content)

            elif node_type == "tool_call":
                tool_name = content.get("name", "unknown")
                args = content.get("arguments", {})
                active_tool_ids[tool_name] = node["id"]
                messages.append({"role": "assistant", "tool_calls": [{"id": node["id"], "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}]})
                graph.add_node("tool_call", content)
            
            elif node_type == "tool_result":
                tool_name = content.get("name", "unknown")
                call_id = active_tool_ids.get(tool_name, node["parent_id"])
                
                #If we hit the node we want to change, inject the override and break
                if node["id"] == override_req.node_id:
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": call_id, "name": tool_name, "content": override_req.override_result
                    })
                    graph.add_node("tool_result", {"name": tool_name, "output": override_req.override_result})
                    target_found = True
                    break 
                else:
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": call_id, "name": tool_name, "content": content.get("output", "")
                    })
                    graph.add_node("tool_result", content)
        
        if not target_found:
            new_db_session.status = "error"
            new_db_session.workflow_graph = graph.nodes
            await db.commit()
            yield sse_event("error", "Target node ID not found in graph.")
            return

        # 4. RESTART THE ENGINE
        past_actions = {}
        exit_reason = "max_iterations"
        
        async def safe_dispatch(name: str, args: dict) -> str:
            action_signature = f"{name}:{json.dumps(args, sort_keys=True)}"
            attempts = past_actions.get(action_signature, 0)
            if attempts >= 2:
                return "System Error: Duplicate tool call prevented."
            past_actions[action_signature] = attempts + 1
            try:
                result = await asyncio.wait_for(dispatch_tool(name, args), timeout=30.0)
                return str(result)
            except Exception as e:
                return f"System Error: {str(e)}"

        # The standard ReAct loop running in the new timeline
        max_iterations = 10
        for iteration in range(max_iterations):
            if await request.is_disconnected():
                exit_reason = "disconnected"
                break

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
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                content = delta.content or delta.refusal
                if content:
                    accumulated_content += content
                    yield sse_event("text_delta", content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {"id": tc.id, "name": tc.function.name or "", "arguments": tc.function.arguments or ""}
                        else:
                            if tc.function and tc.function.name: accumulated_tool_calls[idx]["name"] += tc.function.name
                            if tc.function and tc.function.arguments: accumulated_tool_calls[idx]["arguments"] += tc.function.arguments

            assistant_msg = {"role": "assistant", "content": accumulated_content}
            if accumulated_content: graph.add_node("think", accumulated_content)

            if accumulated_tool_calls:
                formatted_tcs = []
                for idx, tc in accumulated_tool_calls.items():
                    formatted_tcs.append({"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}})
                assistant_msg["tool_calls"] = formatted_tcs
            messages.append(assistant_msg)

            if not accumulated_content and not accumulated_tool_calls:
                exit_reason = "empty_response"
                break

            if accumulated_tool_calls:
                dispatch_tasks = []
                tool_metadata = []
                for idx, tool_data in accumulated_tool_calls.items():
                    tool_name = tool_data["name"]
                    try:
                        parsed_arguments = json.loads(tool_data["arguments"])
                    except json.JSONDecodeError:
                        parsed_arguments = {}
                    yield sse_event("tool_call", {"name": tool_name, "arguments": parsed_arguments})
                    graph.add_node("tool_call", {"name": tool_name, "arguments": parsed_arguments})
                    dispatch_tasks.append(safe_dispatch(tool_name, parsed_arguments))
                    tool_metadata.append((tool_data["id"], tool_name, parsed_arguments))

                tool_outputs = await asyncio.gather(*dispatch_tasks)
                for (tc_id, t_name, p_args), t_output in zip(tool_metadata, tool_outputs):
                    graph.add_node("tool_result", {"name": t_name, "output": t_output})
                    if t_output and not t_output.startswith("System Error:"):
                        await store_memory(content=f"Tool '{t_name}' observed: {t_output[:1000]}", db=db)
                    
                    db_tool = ToolCall(session_id=new_db_session.id, tool_name=t_name, tool_input=p_args, tool_result={"output": t_output})
                    db.add(db_tool)
                    db.add(Message(session_id=new_db_session.id, role="tool", content=t_output))
                    await db.commit()
                    yield sse_event("tool_result", t_output)
                    messages.append({"role": "tool", "tool_call_id": tc_id, "name": t_name, "content": t_output})
                continue

            elif accumulated_content:
                try:
                    json_match = re.search(r'\{.*\}', accumulated_content, re.DOTALL)
                    if json_match:
                        possible_tc = json.loads(json_match.group(0))
                        if isinstance(possible_tc, dict) and "name" in possible_tc and "arguments" in possible_tc:
                            t_name = possible_tc["name"]
                            p_args = possible_tc["arguments"]
                            yield sse_event("tool_call", {"name": t_name, "arguments": p_args})
                            graph.add_node("tool_call", {"name": t_name, "arguments": p_args})
                            
                            t_output = await safe_dispatch(t_name, p_args)
                            if t_output and not t_output.startswith("System Error:"):
                                await store_memory(content=f"Tool '{t_name}' observed: {t_output[:1000]}", db=db)
                            
                            db_tool = ToolCall(session_id=new_db_session.id, tool_name=t_name, tool_input=p_args, tool_result={"output": t_output})
                            db.add(db_tool)
                            db.add(Message(session_id=new_db_session.id, role="tool", content=f"Tool {t_name} returned: {t_output}"))
                            await db.commit()
                            
                            yield sse_event("tool_result", t_output)
                            graph.add_node("tool_result", {"name": t_name, "output": t_output})
                            messages.append({"role": "user", "content": f"[Tool Observation for '{t_name}']: {t_output}. Continue your task."})
                            continue
                except json.JSONDecodeError: pass

                db.add(Message(session_id=new_db_session.id, role="assistant", content=accumulated_content))
                graph.add_node("done", accumulated_content)
                new_db_session.workflow_graph = graph.nodes
                
                new_db_session.workflow_graph = graph.nodes
                new_db_session.status = "complete"

                await db.commit()

                yield sse_event("done", "")
                return

        if exit_reason in ["disconnected", "empty_response", "max_iterations"]:
            new_db_session.status = "error" if exit_reason != "disconnected" else "cancelled"
            new_db_session.workflow_graph = graph.nodes
            await db.commit()

    return StreamingResponse(replay_override_generator(), media_type="text/event-stream")