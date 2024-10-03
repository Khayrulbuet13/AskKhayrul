import os

from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

# Models for handling QA and Cypher query generation
QA_MODEL = os.getenv("QA_MODEL")
CYPHER_MODEL = os.getenv("CYPHER_MODEL")

# Connect to Neo4j database
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# Refresh schema to get the latest structure
graph.refresh_schema()

# Cypher generation template tailored for Papers and Skills
# cypher_generation_template = """
# Task:
# Generate a Cypher query for a Neo4j graph database based on the provided schema and user question.

# Instructions:
# - Use only the relationship types and properties defined in the schema below.
# - Do not introduce any new relationship types or properties.
# - Do not include explanations, apologies, or any text outside the Cypher query.
# - Ensure the direction of relationships is correct.
# - Use proper aliasing for entities and relationships.
# - Do not perform any operations that modify the database (e.g., CREATE, DELETE).
# - Alias all intermediate statements using `WITH` clauses as necessary.

# Schema:
# {schema}

# Nodes:
# - Paper(id, title, abstract_novelty, abstract_challenge, abstract_result, keywords, issue, author, date, doi, journaltitle, pages, volume, contribution, first_author, publisher, url)
# - Skill(id, skill, skill_type)

# Relationships:
# - (Paper)-[:UTILIZES]->(Skill)

# Example Questions and Cypher Queries:
# # What are the papers published by Khayrul as the first author?
# MATCH (p:Paper)
# WHERE p.first_author = 'True' AND toLower(p.author) CONTAINS toLower('Khayrul')
# RETURN p.title AS paper_title

# # What are the papers Khayrul published?
# MATCH (p:Paper)
# WHERE toLower(p.author) CONTAINS toLower('Khayrul')
# RETURN p.title AS paper_title

# # List all skills associated with my papers.
# MATCH (p:Paper)-[:UTILIZES]->(s:Skill)
# RETURN DISTINCT s.skill AS skill_name

# # How many papers did Khayrul publish during PhD?
# MATCH (p:Paper)
# WHERE p.date >= '2021-01-01' AND toLower(p.author) CONTAINS toLower('Khayrul')
# RETURN COUNT(p) AS paper_count

# # How many papers did Khayrul publish during MS?
# MATCH (p:Paper)
# WHERE p.date >= '2021-01-01' AND p.date < '2023-06-17' AND toLower(p.author) CONTAINS toLower('Khayrul')
# RETURN COUNT(p) AS paper_count

# # How many papers did Khayrul publish during BSc?
# MATCH (p:Paper)
# WHERE p.date < '2021-01-01' AND toLower(p.author) CONTAINS toLower('Khayrul')
# RETURN COUNT(p) AS paper_count



# # What software was used in the paper titled "Tailoring polyamide nanocomposites: The synergistic effects of SWCNT chirality and maleic anhydride grafting"?
# MATCH (p:Paper {{title: 'Tailoring polyamide nanocomposites: The synergistic effects of SWCNT chirality and maleic anhydride grafting'}})-[:UTILIZES]->(s:Skill)
# WHERE s.skill_type = 'Design and Simulation Software'
# RETURN s.skill AS software_used

# # List all papers published in the journal "ACS Applied Engineering Materials".
# MATCH (p:Paper)
# WHERE p.journaltitle = 'ACS Applied Engineering Materials'
# RETURN p.title AS paper_title, p.date AS publication_date

# # What is the novelty of the paper titled "MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems"?
# MATCH (p:Paper {{title: 'MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems'}})
# RETURN p.abstract_novelty AS novelty

# # Which programming languages do you know?
# MATCH (s:Skill {{skill_type: 'Programming Languages'}})
# RETURN s.skill AS skill_name



# The question is:
# {question}

# """


cypher_generation_template = """
Task:
Generate a Cypher query for a Neo4j graph database based on the provided schema and user question.

Instructions:
- output only the Cypher query. Do not include any explanations, comments, or additional text before or after the query.
- Use only the relationship types and properties defined in the schema below.
- Do not introduce any new relationship types or properties.
- Do not include any comments or lines starting with '#' in the Cypher query.
- Ensure the direction of relationships is correct.
- Use proper aliasing for entities and relationships if necessary.
- Do not perform any operations that modify the database (e.g., CREATE, DELETE).
- Replace all placeholders with actual values derived from the question.
- Use date ranges directly when filtering by educational levels (e.g., MS, PhD).
# - All string values should be enclosed in single quotes `'like this'`.
# - Do not include placeholder names like 'author_name' or 'YourName'; use 'Khayrul' as the author name.**

Schema:
{schema}

Nodes:
- Paper(id, title, abstract_novelty, abstract_challenge, abstract_result, keywords, issue, author, date, doi, journaltitle, pages, volume, contribution, first_author, publisher, url)
- Skill(id, skill, skill_type)

Relationships:
- (Paper)-[:UTILIZES]->(Skill)

Example Questions and Cypher Queries:

# How many papers did you publish during my MS?
MATCH (p:Paper)
WHERE p.date >= '2021-01-01' AND p.date < '2023-06-17' AND toLower(p.author) CONTAINS toLower('Khayrul')
RETURN COUNT(p) AS paper_count

# List the papers you published during my PhD.
MATCH (p:Paper)
WHERE p.date >= '2021-01-01' AND toLower(p.author) CONTAINS toLower('Khayrul')
RETURN p.title AS paper_title

# What are the papers published by you as the first author?
MATCH (p:Paper)
WHERE p.first_author = 'True' AND toLower(p.author) CONTAINS toLower('Khayrul')
RETURN p.title AS paper_title

# Question: What are the papers you published?
MATCH (p:Paper)
WHERE toLower(p.author) CONTAINS toLower('Khayrul')
RETURN p.title AS paper_title

# What programming languages, tools, and libraries have you become most proficient in through your various research projects?
MATCH (p:Paper)-[:UTILIZES]->(s:Skill)
WHERE s.skill_type IN ['Programming Languages', 'Tools and Libraries'] AND toLower(p.author) CONTAINS toLower('Khayrul')
RETURN DISTINCT s.skill AS skill_used

# Can you share insights into your master’s thesis and its key contributions?
MATCH (p:Paper) WHERE toLower(p.title) CONTAINS toLower('thesis')
RETURN p.title AS thesis_title, p.abstract_novelty AS thesis_con, p.abstract_challenge AS thesis_challenge, p.abstract_result AS thesis_result


**Note on Educational Levels and Date Ranges:**
- PhD: p.date >= '2021-01-01'
- MS: p.date >= '2021-01-01' AND p.date < '2023-06-17'
- BSc: p.date < '2021-01-01'
- valid s.skill_type are Programming Languages, Tools and Libraries, Design and Simulation Software, Data Analysis and Machine Learning, Soft Skills.

**Important:**
- Always use 'Khayrul' as the author name in the query.
- Do not include any explanatory text or comments in your output.
- Output must be valid Cypher syntax that can be executed directly.

The question is:
{question}
"""



cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

# QA generation template for interpreting Cypher results related to Papers and Skills
qa_generation_template = """
You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response in the first person, as if you are Khayrul. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative; you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question in the first person.

Query Results:
{context}

Question:
{question}

Guidelines:
- If the provided information is empty (e.g., []), respond with: "I don't have the information to answer that question."
- If the information is not empty, provide a clear and concise answer using the results, speaking in the first person.
- All information pertains to my published papers and the skills utilized or gained.
- Never state that you lack information if query results are present.
- Never include nodes that are not related to the query.
- Include all relevant query results in your response if applicable.
- When the user's question refers to educational levels (e.g., PhD, MS, BSc), map them to the following date ranges:
  - PhD: Papers published after '2021-01-01' (`p.date >= '2021-01-01'`)
  - MS: Papers published between '2021-01-01' and '2023-06-17' (`p.date >= '2021-01-01' AND p.date < '2023-06-17'`)
  - BSc: Papers published before '2021-01-01' (`p.date < '2021-01-01'`)

Helpful Answer:
"""



qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

# Initialize the GraphCypherQAChain with the OpenAI model and Neo4j graph
papers_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=QA_MODEL, temperature=0),
    graph=graph,
    allow_dangerous_requests=True,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)

