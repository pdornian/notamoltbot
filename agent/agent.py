import requests
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
SYSTEM_PROMPT = """You are a coding help agent. You will help the user write langchain tools
to augment your performance.
"""

model = init_chat_model(
    "claude-haiku-4-5-20251001", temperature=1.0, timeout=10, max_tokens=1000
)

checkpointer = InMemorySaver()

# todo: process this to be a document and have a function to query it.


# this doesn't restrict reading to just markdown files.
@tool
def read_and_store_markdown(url: str, storage_path: Optional[str] = None) -> str:
    """
    Fetches a markdown file from a URL and optionally stores it locally.

    Args:
        url: The URL of the markdown file
        storage_path: Optional local file path to save the markdown.
                     If None, returns content without saving.

    Returns:
        The markdown content as a string
    """
    try:
        # Fetch the content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        markdown_content = response.text

        # Store locally if path provided
        if storage_path:
            with open(storage_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            return (
                f"Successfully fetched and stored markdown from {url} at {storage_path}"
            )

        return markdown_content

    except requests.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except IOError as e:
        return f"Error saving file: {str(e)}"


agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[read_and_store_markdown],
    checkpointer=checkpointer,
)
