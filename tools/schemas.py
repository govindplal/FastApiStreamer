
extract_markdown_from_url = {
    "type": "function",
        "function": {
            "name": "extract_markdown_from_url",
            "description": "Navigates to a URL using a headless browser, renders JavaScript, and returns the webpage content extracted as clean Markdown text. Always use this to read web articles or documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full, valid URL to extract."}
                },
                "required": ["url"]
            }
        },
}

calculate_string_length = {
        "type": "function",
        "function": {
            "name": "calculate_string_length",
            "description": "Calculates the total number of characters in a provided string text block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text string to measure."}
                },
                "required": ["text"]
            }
        }
}

search_memory_tool = {
    "type": "function",
    "function":{
        "name": "search_memory",
        "description": "Searches the agent's long-term memory for past observations, facts, or notes using semantic similarity.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to match against past observations."},
            },
            "required": ["query"]
        }
    }
}

AGENT_TOOLS = [extract_markdown_from_url, calculate_string_length, search_memory_tool]