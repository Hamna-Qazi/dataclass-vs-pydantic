import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated
from pydantic import BaseModel, StringConstraints, field_validator
from agents import Agent, Runner
from agents import OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled
from agents.run import RunConfig


load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")


# 2. Gemini client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


# 3. Pydantic Schema


class BookInfo(BaseModel):
    title: Annotated[str, StringConstraints(min_length=2, max_length=100)]
    author: Annotated[str, StringConstraints(min_length=2, max_length=50)]
    year: int
    @field_validator("title")
    def title_must_not_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Title cannot be empty or just spaces")
        return value.strip()


# 4. Main Async Logic

async def main():
    agent = Agent(
        name="Book Info Extractor",
        instructions="Extract the title, author, publication year, and genre from the text.",
        output_type=BookInfo,  # Now using Pydantic schema
        model=model
    )

    text = """
    The book 'Harry Potter and the Philosopher's Stone' was written by J.K. Rowling in 1997.
    It's a fantasy novel.
    """

    result = await Runner.run(starting_agent=agent, input=text)

    # Structured output (Pydantic object)
    print("\n=== Structured Output ===")
    print(result.final_output)

    # Accessing fields
    print("\n=== Individual Fields ===")
    print("Title:", result.final_output.title)
    print("Author:", result.final_output.author)
    print("Year:", result.final_output.year)
    

    # Convert to dictionary / JSON
    print("\n=== As Dict ===")
    print(result.final_output.model_dump())

    print("\n=== As JSON ===")
    print(result.final_output.model_dump_json(indent=2))



if __name__ == "__main__":
    asyncio.run(main())
