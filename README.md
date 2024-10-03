# AskKhayrul: A RAG-based Chatbot

The **AskKhayrul** chatbot is designed to assist users in exploring Khayrul's research work, publications, and resume details. It provides interactive answers to questions related to Khayrul's published papers, research topics, and background, leveraging state-of-the-art technology.

This chatbot is powered by a **LangChain agent**, utilizing **retrieval-augmented generation (RAG)** for comprehensive and accurate answers. The RAG framework integrates both structured data (managed via **Neo4j**) and unstructured data (stored in **ChromaDB**), ensuring high-quality responses to a wide range of queries. The system is fully containerized using **Docker** for seamless deployment and scalability.

## Key Features
- **Interactive Q&A**: Engage with Khayrul's research and publications using a user-friendly interface.
- **RAG Framework**: Combines structured and unstructured data for more accurate responses.
- **Dockerized Deployment**: Ensures easy deployment and scalability for different environments.

## File Structure
For better organization and easy navigation, the project is structured as follows:

```
source/
├── .env
├── .gitignore
├── LICENSE.txt
├── README.md
├── chatbot_api/
│   ├── Dockerfile
│   ├── pdfs/
│   │   ├── 1.pdf
│   │   ├── 2.pdf
│   │   ├── ...
│   │   ├── ...
│   ├── requirements.txt
│   └── src/
│       ├── agents/
│       │   └── chatbot_rag_agent.py
│       ├── chains/
│       │   ├── education_chain.py
│       │   └── papers_chain.py
│       ├── entrypoint.sh
│       ├── generate_vector_store.py
│       ├── main.py
│       ├── models/
│       │   └── chatbot_rag_query.py
│       ├── tools/
│       │   └── papersQA.py
│       └── utils/
│           └── async_utils.py
├── chatbot_frontend/
│   ├── Dockerfile
│   ├── image/
│   │   ├── hi.png
│   │   └── khayrul.png
│   ├── pyproject.toml
│   └── src/
│       ├── entrypoint.sh
│       └── main.py
├── chatbot_neo4j_etl/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── src/
│       ├── chroma_db.py
│       ├── education_csv_write.py
│       └── entrypoint.sh
├── data/
│   ├── courses.csv
│   ├── education.csv
│   ├── papers.json
│   └── skills.csv
├── docker-compose.yml
├── requirements.txt
├── tests/
│   ├── async_agent_requests.py
│   └── sync_agent_requests.py
└── tree.txt  

```

## Getting Started with AskKhayrul

Follow this guide to set up the **AskKhayrul** chatbot locally:

### Install Dependencies
To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Khayrulbuet13/AskKhayrul.git
cd AskKhayrul
pip install -r requirements.txt
```

### Running the Application
The chatbot service can be launched using **Docker Compose**:

```bash
docker-compose up
```

Ensure that both the **API** and **frontend** services are running. The chatbot will be accessible via the defined frontend URL.

## Built With
This project is built using the following tools and libraries:

- **LangChain**: Core agent framework for retrieval-augmented generation.
- **Neo4j**: Graph database for managing structured research data.
- **ChromaDB**: Vector store for handling unstructured data.
- **Docker**: Containerization for portability and scalability.
- **FastAPI**: Backend framework for serving the chatbot API.
- **Streamlit**: Frontend for interactive user queries.

## License
This project is distributed under the MIT License. See the `LICENSE` file for more details.

## Contact
**Khayrul Islam**  
- LinkedIn: [@LinkedIN](https://linkedin.com/in/Khayrulbuet13)  
- Email: [khayrulbuet13@alum.lehigh.edu](mailto:khayrulbuet13@alum.lehigh.edu)

Project Link: [AskKhayrul GitHub Repo](https://github.com/Khayrulbuet13/AskKhayrul) 

---

