import time

import requests

CHATBOT_URL = "http://localhost:8000/chatbot-rag-agent"

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
outputs = [requests.post(CHATBOT_URL, json=data) for data in request_bodies]
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")
