import os
import json
from dotenv import load_dotenv
from groq import Groq
from webscout import DeepWEBS


def generate(user_prompt, system_prompt="Be Short and Concise", prints=False) -> str:
    """
    Generates a response to the user's prompt using the Groq API and DeepWEBS for search.

    Parameters:
    - user_prompt (str): The user's input prompt.
    - system_prompt (str): System's instruction to the model.
    - prints (bool): Flag to enable printing of the process.

    Returns:
    - str: The final response generated by the model.
    """

    def deepwebs_search(query, max_results=10):
        """Performs a web search using DeepWEBS and returns results as JSON."""
        deepwebs = DeepWEBS()
        search_config = DeepWEBS.DeepSearch(
            queries=[query],
            max_results=max_results,
            extract_webpage=False,     # Extract webpage content
            safe=False,                # Enable SafeSearch
            types=["web"],             # Types of search results (web, image, videos, news)
            overwrite_query_html=True,  # Overwrite existing query HTML files
            overwrite_webpage_html=True, # Overwrite existing webpage HTML files
        )
        search_results = deepwebs.queries_to_search_results(search_config)
        return json.dumps(search_results)

    # Define available tools with function descriptions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "deepwebs_search",
                "description": "Performs a web search using DeepWEBS",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search on the web",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results (default: 5)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    api_key = "ADD-OWN-KEY"

    # Initial response from Groq
    response = Groq(api_key=api_key).chat.completions.create(
        model='llama3-70b-8192',
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )
    response_message = response.choices[0].message
    if prints:
        print(f"Initial Response: {response_message} \n")

    # Handle tool calls
    tool_calls = response_message.tool_calls
    if tool_calls:
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "deepwebs_search":
                function_response = deepwebs_search(**function_args)
            else:
                # Handle other potential functions here
                raise NotImplementedError(f"Function '{function_name}' not implemented.")

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        # Final response from Groq
        second_response = Groq(api_key=api_key).chat.completions.create(
            model='llama3-70b-8192',
            messages=messages
        )
        return second_response.choices[0].message.content

    else:
        return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage:
    user_query = "Search the web for GPT-5 release date and features"
    response = generate(user_prompt=user_query, prints=False)
    print(response)
