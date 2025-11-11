from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from os import environ
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from langchain_core.prompts import ChatPromptTemplate
from enum import Enum
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(model = environ.get("CHAT_MODEL_NAME", ""), api_key = environ.get("OPENAI_API_KEY", ""), temperature = 0.3, 
                 seed = 24288, top_p = 0.2)

class SentimentEnum(str, Enum):
    happy = "happy"
    neutral = "neutral"
    sad = "sad"

class AggressivenessEnum(int, Enum):
    low = 1
    medium_low = 2
    medium = 3
    medium_high = 4
    high = 5

class LanguageEnum(str, Enum):
    spanish = "spanish"
    english = "english"
    french = "french"
    german = "german"
    italian = "italian"

class Classification(BaseModel):
    sentiment : SentimentEnum = Field(..., description = "The sentiment of the text")
    aggressiveness : AggressivenessEnum = Field(..., description = "Describes how aggressive the statement is; the higher the number, the more aggressive")
    language : LanguageEnum = Field(..., description = "The language the text is written in")


system_prompt = """
    You are an expert text classifier. Analyze the user's message and provide a JSON response 
    with the keys: sentiment (happy, neutral, sad), aggressiveness (integer from 1 to 5), 
    and language (one of spanish, english, french, german, italian). 
    Respond ONLY with a valid JSON matching the Classification schema.
"""


def chat_conversation(llm_object : ChatOpenAI, system_prompt : str, user_message : str):
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{user_input}")])
    messages = prompt.format_messages(user_input = user_message)
    structured_llm = llm_object.with_structured_output(Classification)
    response = structured_llm.invoke(messages)
    response_data = response.model_dump()
    return response_data

def main():
    console = Console()
    console.print(Panel("[bold cyan]Welcome to the Sentiment Analysis App[/bold cyan]", expand = False))
    
    user_message = ""
    while not user_message.strip():
        user_message = Prompt.ask("[bold yellow]Enter text to analyze sentiment[/bold yellow]")
        if not user_message.strip():
            console.print("[bold red]Input cannot be empty. Please enter some text.[/bold red]")
    
    result = chat_conversation(llm, system_prompt, user_message)
    
    output_panel = Panel(f"""
                         [bold green]Sentiment: [/bold green] {result["sentiment"].value}
                         [bold green]Aggressiveness: [/bold green] {result["aggressiveness"].value}
                         [bold green]Language: [/bold green] {result["language"].value}
                         """,
                         title = "[bold magenta]Classification Result[/bold magenta]",
                         expand = False
                         )
    
    console.print(output_panel)

if __name__ == "__main__":
    main()