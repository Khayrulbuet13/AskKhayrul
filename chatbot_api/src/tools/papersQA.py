import os
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain import hub

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory="/app/vector_store",
    embedding_function=embeddings
)

# Initialize the chat model
chat_model = ChatOpenAI(
    model=os.getenv("AGENT_MODEL"),
    temperature=0,
)

# Create the RetrievalQA chain for papers
papers_qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
)
