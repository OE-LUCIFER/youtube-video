from getpass import getpass
import webscout.AIutel
from webscout import DeepWEBS # Import the main Webscout library
import click
import sys
from rich.console import Console
from rich.markdown import Markdown

# Provider map for dynamic model selection
provider_map = {
    "phind": webscout.PhindSearch,
    "opengpt": webscout.OPENGPT,
    "koboldai": webscout.KOBOLDAI,
    "blackboxai": webscout.BLACKBOXAI,
    "llama2": webscout.LLAMA2,
    "yepchat": webscout.YEPCHAT,
    "leo": webscout.LEO,
    "groq": webscout.GROQ,
    "openai": webscout.OPENAI,
    "perplexity": webscout.PERPLEXITY,
    "you": webscout.YouChat,
    "xjai": webscout.Xjai,
    "cohere": webscout.Cohere,
    "reka": webscout.REKA,
    "thinkany": webscout.ThinkAnyAI,
}


def main():
    max_tokens = 600
    temperature = 0.2
    top_k = -1
    top_p = 0.999
    model = "Phind Model"  # Use Phind Model as default
    auth = None  # No authentication needed for Phind
    timeout = 30
    is_conversation = True  # Disable/Enable conversation history
    filepath = None  # No history file
    update_file = True
    intro = None
    history_offset = 10250
    awesome_prompt = None
    proxy_path = None
    provider = "phind"  # Default provider
    quiet = False  # Suppress output
    chat_completion = False
    ignore_working = False
    rawdog = True  # Enable Rawdog
    internal_exec = False  # Internal script execution
    confirm_script = False  # Don't ask for confirmation before execution
    interpreter = "python"
    optimizer = "code"  # Always use the 'code' optimizer
    websearch = False  # beta feature
    read_aloud = False
    read_aloud_voice = "Brian"

    # Initialize DeepWEBS for web search
    D = DeepWEBS()

    # Check if Rawdog mode is enabled
    if rawdog:
        # Initialize Rawdog
        rawdog = webscout.AIutel.RawDog(
            quiet=quiet,
            internal_exec=internal_exec,
            confirm_script=confirm_script,
            interpreter=interpreter,
        )
        # Set intro to Rawdog's default system prompt
        intro = rawdog.intro_prompt
        getpass.getuser = lambda: "RawDog"

    # Initialize the AI model based on the provider
    bot = provider_map[provider](
        is_conversation=is_conversation,
        max_tokens=max_tokens,
        timeout=timeout,
        intro=intro,  # Use Rawdog's intro if enabled
        filepath=filepath,
        update_file=update_file,
        proxies={},  # Empty proxies
        history_offset=history_offset,
        act=awesome_prompt,
        model=model,
        quiet=quiet,
    )

    # Initialize Rich console for prettified output
    console = Console()

    while True:
        # Get user input
        user_prompt = input("You: ")

        # Perform web search and modify prompt if websearch mode is enabled
        if websearch:
            search_params = D.DeepSearch(
                queries=[user_prompt], # Query to search
                result_num=5, # Number of search results
                safe=True, # Enable SafeSearch
                types=["web"], # Search type: web
                extract_webpage=True, # True for extracting webpages
                overwrite_query_html=False,
                overwrite_webpage_html=False,
            )
            
            try:
                search_results = D.queries_to_search_results(search_params)
                if search_results:
                    # Extract and format results
                    formatted_results = "\n".join(
                        f"Result {i+1}: {result['title']} - {result['url']}\n\nBody: {result.get('text', '')}" 
                        for i, result in enumerate(search_results[0]['query_results'])
                    )
                    user_prompt += f"\n\n## Web Search Results are:\n\n{formatted_results}"
            except Exception as e:
                console.print(Markdown(f"[red]Error during web search: {e}[/red]"))
                continue

       # Apply the optimizer to the prompt
        user_prompt = webscout.AIutel.Optimizers.code(user_prompt)

        # Generate response using the AI model
        try:
            response = bot.chat(user_prompt)
        except webscout.exceptions.FailedToGenerateResponseError as e:
            console.print(Markdown(f"LLM: [red]{e}[/red]"))
            continue

        # Check if Rawdog mode is enabled
        if rawdog:
            # Process and execute the generated script using Rawdog
            try:
                is_feedback = rawdog.main(response)
            except Exception as e:
                console.print(Markdown(f"LLM: [red]Error: {e}[/red]"))
                continue
            if is_feedback:
                # If there's feedback from Rawdog, continue the conversation
                console.print(Markdown(f"LLM: {is_feedback}"))
                continue
            # If no feedback, continue with a new prompt
            console.print(Markdown("LLM: (Script executed successfully)")) 
        else:
            # If Rawdog is not enabled, simply print the response
            console.print(Markdown(f"LLM: {response}")) 

        if read_aloud:
            # Read the response aloud using text-to-speech
            try:
                audio_content = webscout.AIutel.Audio.text_to_audio(
                    response, voice=read_aloud_voice, auto=True
                )
                webscout.AIutel.Audio.play(audio_content)
            except Exception as e:
                console.print(
                    Markdown(f"[red]Error during text-to-speech: {e}[/red]")
                )

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        click.secho("\nExiting...", fg="yellow")
        sys.exit(0) 