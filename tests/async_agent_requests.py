import asyncio
import time

import httpx

CHATBOT_URL = "http://localhost:8000/chatbot-rag-agent"


async def make_async_post(url, data):
    timeout = httpx.Timeout(timeout=120)
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, timeout=timeout)
        return response


async def make_bulk_requests(url, data):
    tasks = [make_async_post(url, payload) for payload in data]
    responses = await asyncio.gather(*tasks)
    outputs = [r.json()["output"] for r in responses]
    return outputs


questions = [
    "Could you provide an overview of your educational background?",
    "Can you elaborate on the methodologies and techniques you employed in your recent publication on 'Multiplex Image Machine Learning'?",
    "Could you explain the main concept and significance of your paper titled 'Multiplex Image Machine Learning'?",
    "Can you share insights into your master’s thesis and its key contributions?",
    "Tell me about your research during your Ph.D.?",
    "Did your coursework include any subjects related to machine learning?",
    "Where did you complete your master's degree, and what was the focus of your studies?",
]


request_bodies = [{"text": q} for q in questions]

start_time = time.perf_counter()
outputs = asyncio.run(make_bulk_requests(CHATBOT_URL, request_bodies))
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")
