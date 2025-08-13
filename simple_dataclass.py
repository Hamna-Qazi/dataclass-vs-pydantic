import os
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from agents import Agent, Runner
from agents import OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from agents.run import RunConfig


load_dotenv()
set_tracing_disabled(True)
# Get the Gemini API key from environment variables
gemini_api_key=os.getenv("GEMINI_API_KEY")

# Raise error if the Gemini API key is missing
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Create an external OpenAI client to connect to Gemini API
external_client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the chat completion model using Gemini
model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
# Configuration for running the agent
config=RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True 
)

# Dataclass schema for structured output
@dataclass
class BookInfo:
    title:str
    author:str
    year:int

async def main():
    agent=Agent(
        name="Book Info Extractor",
        instructions="Extract the title, author, and publication year from the text.",
        output_type=BookInfo,
        model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=external_client
    ),
)
    
    text = "The book 'Harry Potter and the Philosopher's Stone' was written by J.K. Rowling in 1997."
    result =await Runner.run(starting_agent=agent, input=text)


    # Print structured output
    print(result.final_output)
    print(result.final_output.title)
    print(result.final_output.author)
    print(result.final_output.year)

if __name__ == "__main__":
    asyncio.run(main())