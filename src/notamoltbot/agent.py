import requests
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langchain.tools import tool
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path, PurePosixPath


load_dotenv()
SYSTEM_PROMPT = """You are a coding help agent. You will help the user write langchain tools
to augment your performance.
"""

model = init_chat_model(
    "claude-haiku-4-5-20251001", temperature=1.0, timeout=10, max_tokens=1000
)

checkpointer = InMemorySaver()


# this doesn't restrict reading to just markdown files.
@tool
def read_and_store_web_markdown(url: str, storage_path: Optional[str] = None) -> str:
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


def read_file(filepath: str) -> str:
    """
    Fetches a local file and reads its text.

    Args:
        filepath: The path of the file

    Returns:
        The file content as a string
    """
    try:
        # Store locally if path provided

        with open(filepath, "r", encoding="utf-8") as f:
            file_content_str = f.read()
        return file_content_str

    except OSError as e:
        return f"File not found: {str(e)}"


skill_files = {}
# for root, dirs, files in os.walk("skills"):
#     for name in files:
#         # add leading slash or this explodes
#         skill_filepath = f"/{os.path.join(root, "/", name)}"
#         skill_files[skill_filepath] = create_file_data(read_file(skill_filepath))

for root, dirs, files in Path("skills").walk():
    for name in files:
        # add leading slash or this explodes
        skill_filepath = PurePosixPath(root).joinpath(name)
        content = read_file(skill_filepath)
        skill_files[f"/{skill_filepath}"] = create_file_data(content)

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    skills=["/skills/"],
    tools=[read_and_store_web_markdown],
    checkpointer=checkpointer,
)
