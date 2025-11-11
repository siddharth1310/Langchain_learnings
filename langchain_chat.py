#-----------------------------------------EXAMPLE - LLM CONVERSATION WITHOUT THE USE OF ChatPromptTemplate (Langchain's Inbuilt method)

# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from os import environ
# from rich.console import Console
# from rich.panel import Panel
# from rich.prompt import Prompt

# load_dotenv()

# llm = ChatOpenAI(model = environ.get("CHAT_MODEL_NAME", ""), 
#                  api_key = environ.get("OPENAI_API_KEY", ""), 
#                  temperature = 0.3,
#                  seed = 24288,
#                  top_p = 0.2
#                  )

# system_prompt = "You are a well known translator. Your job is to convert the user's message into Russian."

# def chat_conversation(llm_object : ChatOpenAI, system_prompt : str, user_message : str):
#     messages = [("system", system_prompt), ("human", user_message)]
#     return llm_object.invoke(messages)

# def main():
#     console = Console()
#     console.print(Panel("[bold cyan]Welcome to the Translator App[/bold cyan]", expand = False))
    
#     user_message = ""
    
#     while not user_message.strip():
#         user_message = Prompt.ask("[bold yellow]Enter text to translate to Russian[/bold yellow]")
#         if not user_message.strip():
#             console.print("[bold red]Input cannot be empty. Please enter some text.[/bold red]")
    
#     result = chat_conversation(llm, system_prompt, user_message)
    
#     console.print(Panel(result.content, title = "[bold green]Translation[/bold green]", expand = False))

# if __name__ == "__main__":
#     main()

#-----------------------------------------EXAMPLE - LLM CONVERSATION WITH THE USE OF ChatPromptTemplate (Langchain's Inbuilt method)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from os import environ
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model = environ.get("CHAT_MODEL_NAME", ""), 
                 api_key = environ.get("OPENAI_API_KEY", ""), 
                 temperature = 0.3,
                 seed = 24288,
                 top_p = 0.2
                 )

system_prompt = "You are a well known translator. Your job is to convert the user's message from {language1} into {language2}."

def chat_conversation(llm_object : ChatOpenAI, system_prompt : str, user_message : str, language_from : str, language_to : str):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{user_input}")])
    messages = prompt.format_messages(language1 = language_from, language2 = language_to, user_input = user_message)
    return llm_object.invoke(messages)

def main():
    console = Console()
    console.print(Panel("[bold cyan]Welcome to the Translator App[/bold cyan]", expand = False))
    
    language_from = ""
    while not language_from.strip():
        language_from = Prompt.ask("[bold yellow]Enter the language to translate from[/bold yellow]")
        if not language_from.strip():
            console.print("[bold red]Input cannot be empty. Please enter the source language.[/bold red]")
    
    language_to = ""
    while not language_to.strip():
        language_to = Prompt.ask("[bold yellow]Enter the language to translate to[/bold yellow]")
        if not language_to.strip():
            console.print("[bold red]Input cannot be empty. Please enter the target language.[/bold red]")
    
    user_message = ""
    while not user_message.strip():
        user_message = Prompt.ask(f"[bold yellow]Enter the text to translate from {language_from} to {language_to}[/bold yellow]")
        if not user_message.strip():
            console.print("[bold red]Input cannot be empty. Please enter some text.[/bold red]")
    
    result = chat_conversation(llm, system_prompt, user_message, language_from, language_to)
    
    console.print(Panel(result.content, title = "[bold green]Translation[/bold green]", expand = False))


if __name__ == "__main__":
    main()