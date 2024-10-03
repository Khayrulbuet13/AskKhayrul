from agents.chatbot_rag_agent import chatbot_rag_agent_executor
from fastapi import FastAPI
from models.chatbot_rag_query import ChatbotQueryInput, ChatbotQueryOutput
from utils.async_utils import async_retry

import os
import httpx
SLACK_WEBHOOK_URL = os.environ.get('SLACK_WEBHOOK_URL')

app = FastAPI(
    title="AskKhayrul Chatbot",
    description="Endpoints for a AskKhayrul RAG chatbot",
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await chatbot_rag_agent_executor.ainvoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


async def send_message_to_slack(question: str, answer: str):
    if not SLACK_WEBHOOK_URL:
        return
    message = {
        "text": f"*User asked:*\n{question}\n\n*Answer:*\n{answer}"
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(SLACK_WEBHOOK_URL, json=message)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        # Handle exception (e.g., log it)
        print(f"Error sending message to Slack: {exc}")


@app.post("/chatbot-rag-agent")
async def query_chatbot_agent(
    query: ChatbotQueryInput,
) -> ChatbotQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    # Send the question and answer to Slack
    await send_message_to_slack(query.text, query_response["output"])

    return query_response
