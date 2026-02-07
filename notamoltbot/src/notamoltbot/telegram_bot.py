import logging
import os
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    filters,
    MessageHandler,
)
from agent import agent

from dotenv import load_dotenv

load_dotenv()

token = os.getenv("TELEGRAM_BOT_KEY")

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a bot that eats Claude tokens. Please talk to me!",
    )


config = {"configurable": {"thread_id": "1"}}


# Handle messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages and generate responses using Langchain."""
    user_message = {"messages": [{"role": "user", "content": update.message.text}]}
    response = agent.invoke(user_message, config=config)["messages"][-1]
    print(response)
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=response.content
    )


def main():
    application = ApplicationBuilder().token(token).build()
    start_handler = CommandHandler("start", start)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(start_handler)
    application.add_handler(message_handler)
    application.run_polling()


if __name__ == "__main__":
    main()
