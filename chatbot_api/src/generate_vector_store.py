import os
import asyncio
import warnings
from neo4j import GraphDatabase
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables from .env file
# load_dotenv()

warnings.filterwarnings("ignore")
from utils.async_utils import async_retry

class DocumentVectorizer:
    def __init__(self, embeddings, vector_store_path="vector_store", distance_metric='cosine'):
        self.vector_store_path = vector_store_path
        self.embeddings = embeddings
        self.distance_metric = distance_metric
        # self.neo4j_uri = os.getenv("NEO4J_URI")
        # self.neo4j_user = os.getenv("NEO4J_USER")
        # self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.neo4j_uri = 'neo4j+s://448e074a.databases.neo4j.io'
        self.neo4j_user = 'neo4j'
        self.neo4j_password = 'rH3HeRmKE7z2Y4vnHIxO60ZCIZEm9pO_6hd8TwStVEM'

    def connect_to_neo4j(self):
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

    def close(self):
        if self.driver is not None:
            self.driver.close()

    @async_retry(max_retries=5, delay=2)
    async def fetch_metadata(self):
        with self.driver.session(database="neo4j") as session:
            query = """
            MATCH (p:Paper)
            OPTIONAL MATCH (p)-[:UTILIZES]->(s:Skill)
            WITH p, collect(s.skill) AS skills
            RETURN p.id AS id, properties(p) AS properties, skills
            """
            result = session.run(query)
            id_to_metadata = {}
            for record in result:
                properties = record["properties"]
                skills_str = ', '.join(record["skills"]) if record["skills"] else "No skills listed"
                properties["skills"] = skills_str
        return id_to_metadata
    

    def load_pdfs(self, path):
        if os.path.isdir(path):
            loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
        elif os.path.isfile(path) and path.endswith('.pdf'):
            loader = PyPDFLoader(path)
        else:
            raise ValueError("Unsupported file format or path. Please provide a valid PDF file or directory containing PDFs.")
        documents = loader.load()
        return documents
    

    async def create_vector_store(self, path):
        self.connect_to_neo4j()

        # Await the fetch_metadata function since it's asynchronous
        id_to_metadata = await self.fetch_metadata()

        documents = self.load_pdfs(path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(documents)

        for doc in text_chunks:
            source_path = doc.metadata.get('source', '')
            filename = os.path.basename(source_path)
            file_id, _ = os.path.splitext(filename)
            json_meta = id_to_metadata.get(file_id, {})
            doc.metadata.update(json_meta)

        text_contents = [doc.page_content for doc in text_chunks]
        metadatas = [doc.metadata for doc in text_chunks]

        collection_metadata = {'hnsw:space': self.distance_metric}
        Chroma.from_texts(
            text_contents,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_metadata=collection_metadata,
            persist_directory=self.vector_store_path
        )

        # Close the Neo4j driver
        self.close()
        print("Vector store created successfully.")





    def retrieve_documents(self, query, k=3, threshold=0.5):
        vector_db = Chroma(persist_directory=self.vector_store_path, embedding_function=self.embeddings)
        retrieved_documents = vector_db.similarity_search_with_score(query, k=k)

        filtered_docs_by_threshold = [doc for doc in retrieved_documents if doc[1] > threshold]
        relevant_docs = sorted(filtered_docs_by_threshold, key=lambda x: x[1], reverse=True)

        sources = [{'score': round(score, 3), 'document': doc.page_content, 'metadata': doc.metadata} for doc, score in relevant_docs]
        return sources

if __name__ == "__main__":
    embedding = OpenAIEmbeddings()  # Use OpenAI embeddings
    vectorizer = DocumentVectorizer(
        embeddings=embedding,
        vector_store_path="/app/vector_store"
        # vector_store_path="/home/mdi220/Simulations/Git_repository/RAG-REALPYTHON/chatbot_docker/chatbot_api/vector_store",
    )

    # Run the async function in the event loop
    asyncio.run(
        vectorizer.create_vector_store("/app/pdfs")
        # vectorizer.create_vector_store("/home/mdi220/Simulations/Git_repository/RAG-REALPYTHON/chatbot_docker/chatbot_api/pdfs")
    )
