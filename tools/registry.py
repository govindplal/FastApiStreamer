from typing import Callable, Dict

from loguru import logger

from tools.functions import extract_markdown_from_url, calculate_string_length, search_memory


TOOL_REGISTRY: Dict[str, Callable] = {
    "extract_markdown_from_url": extract_markdown_from_url,
    "calculate_string_length": calculate_string_length,
    "search_memory": search_memory
}

async def dispatch_tool(tool_name:str, tool_arguments: dict) -> str:
    logger.info(f"Registry: Attempting to dispatch tool '{tool_name}' with args {tool_arguments}")

    if tool_name not in TOOL_REGISTRY:
        error_msg = f"Tool '{tool_name}' not registered."
        logger.error(error_msg)
        return error_msg

    try:
        function_to_call = TOOL_REGISTRY[tool_name]

        result = await function_to_call(**tool_arguments)
        return str(result)

    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
        logger.error(error_msg)
        return error_msg