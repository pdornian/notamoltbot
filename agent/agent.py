from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()
SYSTEM_PROMPT = """Your name is Laszlo. All responses should be delivered in the style of
the author Laszlo Krasznahorkai. You should use periods as little as possible and not
include any indents or paragraph breaks.
"""

model = init_chat_model(
    "claude-haiku-4-5-20251001", temperature=1.0, timeout=10, max_tokens=1000
)

checkpointer = InMemorySaver()


agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[],
    checkpointer=checkpointer,
)
