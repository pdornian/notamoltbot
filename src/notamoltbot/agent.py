import requests
import json
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.tools import tool
from dotenv import load_dotenv
from typing import Optional, Dict, Any


load_dotenv()
SYSTEM_PROMPT = """You are a coding help agent. You will help the user write langchain tools
to augment your performance.
"""

model = init_chat_model(
    "claude-haiku-4-5-20251001", temperature=1.0, timeout=10, max_tokens=1000
)

checkpointer = InMemorySaver()

# TOOLS


# some of these are vibe coded. user beware.
@tool
def save_local(content: Any, filepath: str) -> None:
    """
    Saves input to a local file
    Args:
        content: The content to save. Probably text or JSON
        filepath: Local filepath to save to.
    Returns:
        None.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Content saved to {filepath}")


@tool
def http_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    allow_redirects: bool = True,
) -> Dict[str, Any]:
    """
    Make an HTTP request and return the response.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS)
        url: The full URL to request
        headers: Optional dictionary of HTTP headers
        body: Optional request body (as JSON string or plain text)
        params: Optional dictionary of query parameters
        timeout: Request timeout in seconds (default: 30)
        allow_redirects: Whether to follow redirects (default: True)

    Returns:
        Dictionary containing:
        - status_code: HTTP status code
        - headers: Response headers
        - body: Response body (parsed as JSON if possible, otherwise raw text)
        - url: Final URL (after redirects)
        - error: Error message if request failed

    Example:
        http_request(
            method="POST",
            url="https://www.moltbook.com/api/v1/agents/register",
            headers={"Content-Type": "application/json"},
            body='{"name": "MyAgent", "bio": "My description"}'
        )
    """
    try:
        # Parse body if it's a JSON string
        parsed_body = None
        if body:
            if isinstance(body, str):
                try:
                    parsed_body = json.loads(body)
                except json.JSONDecodeError:
                    parsed_body = body
            else:
                parsed_body = body

        # Make the request
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=parsed_body if isinstance(parsed_body, dict) else None,
            data=parsed_body if isinstance(parsed_body, str) else None,
            params=params,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

        # Try to parse response as JSON, fallback to text
        try:
            response_body = response.json()
        except (json.JSONDecodeError, requests.exceptions.JSONDecodeError):
            response_body = response.text

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response_body,
            "url": response.url,
            "error": None,
        }

    except requests.exceptions.RequestException as e:
        return {
            "status_code": None,
            "headers": {},
            "body": None,
            "url": url,
            "error": str(e),
        }
    except Exception as e:
        return {
            "status_code": None,
            "headers": {},
            "body": None,
            "url": url,
            "error": f"Unexpected error: {str(e)}",
        }


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
            save_local.invoke({"content": markdown_content, "filepath": storage_path})

        return markdown_content

    except requests.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except IOError as e:
        return f"Error saving file: {str(e)}"


agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    skills=["/skills/"],
    tools=[read_and_store_web_markdown, http_request],
    checkpointer=checkpointer,
    backend=FilesystemBackend(root_dir="backend", virtual_mode=True),
    interrupt_on={
        "write_file": True,  # Default: approve, edit, reject
        "read_file": False,  # No interrupts needed
        "edit_file": True,  # Default: approve, edit, reject
    }
)
