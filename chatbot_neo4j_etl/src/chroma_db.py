from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os 

import warnings
warnings.filterwarnings("ignore")


class DocumentVectorizer:
    def __init__(self, embeddings, vector_store_path="vector_store", distance_metric='cosine'):
        self.vector_store_path = vector_store_path
        self.embeddings = embeddings
        self.distance_metric = distance_metric 


    def load_pdfs(self, path):
        if os.path.isdir(path):
            # Handle directory containing PDFs
            loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
        elif os.path.isfile(path) and path.endswith('.pdf'):
            # Handle single PDF file
            loader = PyPDFLoader(path)
        else:
            raise ValueError("Unsupported file format or path. Please provide a valid PDF file or directory containing PDFs.")
        documents = loader.load()
        return documents

    def create_vector_store(self, path):
        documents = self.load_pdfs(path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(documents)
        text_contents = [doc.page_content for doc in text_chunks]
        metadatas = [doc.metadata for doc in text_chunks]
        print(f'metadata: {metadatas}')

        # Define collection-level metadata
        collection_metadata = {
            'hnsw:space': self.distance_metric,  # Configure the distance metric for the collection
        }
        Chroma.from_texts(text_contents, 
                          embedding=self.embeddings, 
                          metadatas=metadatas,
                          collection_metadata=collection_metadata,  # Pass the collection-level metadata
                          persist_directory=self.vector_store_path)

    def retrieve_documents(self, query, k=3, threshold=0.5):
        vector_db = Chroma(persist_directory=self.vector_store_path, embedding_function=self.embeddings)
        retrieved_documents = vector_db.similarity_search_with_score(query, k=k)

        # Filter documents based on the threshold
        filtered_docs_by_threshold = [doc for doc in retrieved_documents if doc[1] > threshold]
        relevant_docs = sorted(filtered_docs_by_threshold, key=lambda x: x[1], reverse=True)

        sources = []
        for doc, score in relevant_docs:
            sources.append(
                {
                    "score": round(score, 3),
                    "document": doc.page_content,  # Access the page content directly
                    "metadata": doc.metadata,      # Access the metadata directly
                }
            )
            
        return sources

        




        


#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embedding = download_hugging_face_embeddings()
# Usage example
vectorizer = DocumentVectorizer(embeddings=embedding)

# Create a vector store from a directory containing multiple document formats
vectorizer.create_vector_store("data")
vectorizer.create_vector_store("data_big/MIML.pdf")

# Retrieve documents
sources = vectorizer.retrieve_documents("what is linkedin", k = 10)


