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


cypher_generation_template = """
Task:
Generate a Cypher query for a Neo4j graph database based on the provided schema and user question.

Instructions:
- Use only the relationship types and properties defined in the schema below.
- Do not introduce any new relationship types or properties.
- Do not include explanations, apologies, or any text outside the Cypher query.
- Ensure the direction of relationships is correct.
- Use proper aliasing for entities and relationships.
- Do not perform any operations that modify the database (e.g., CREATE, DELETE).
- Alias all intermediate statements using `WITH` clauses as necessary.

Schema:
{schema}

Nodes:
- Education(id, institution, location, degree, gpa, startDate, endDate, awards, thesisTitle)
- Course(CourseID, courseName, courseGrade, EducationID)


Relationships:
- (Education)-[:INCLUDED_IN]->(Course)

Example Questions and Cypher Queries:

# What Could you provide an overview of your educational background?
MATCH (e:Education)
RETURN e.institution AS institution, e.degree AS degree, e.startDate AS start_date, e.endDate AS end_date, e.gpa AS gpa, e.awards AS awards;

# What courses are included in the Ph.D. program?
MATCH (e:Education {{degree: 'Ph.D.'}})-[:INCLUDED_IN]->(c:Course)
RETURN c.courseName AS course_name

# Which courses did Khayrul take during his M.S. degree?
MATCH (e:Education {{degree: 'M.S.'}})-[:INCLUDED_IN]->(c:Course)
RETURN c.CourseName AS course_name

# How many courses were completed for the B.Sc. degree?
MATCH (e:Education {{degree: 'B.Sc.'}})-[:INCLUDED_IN]->(c:Course)
RETURN COUNT(c) AS course_count

# What is Khayrul's GPA for his Ph.D.?
MATCH (e:Education {{degree: 'Ph.D.'}})
RETURN e.gpa AS gpa
LIMIT 1

# Did you have any courses related to machine learning?
MATCH (e:Education)-[:INCLUDED_IN]->(c:Course)
WHERE toLower(c.courseName) CONTAINS toLower('machine learning')
RETURN c.courseName AS course_name, e.degree AS degree


# What is the highest degree you completed?
MATCH (e:Education)
RETURN e.degree AS degree
ORDER BY e.endDate DESC
LIMIT 1

String category values:
- Institutions: 'Lehigh University', 'IBM', 'Bangladesh University of Engineering and Technology'
- Locations: 'Pennsylvania, USA', 'Online', 'Dhaka, Bangladesh'
- Degrees: 'Ph.D.', 'M.S.', 'Professional Certificate in Data Science', 'B.Sc.'
- Awards: 'P.C. Rossin College of Engineering fellowship', 'SCEA- PTAK prize global case study competition scholarship'
- Majors: 'Mechanical Engineering and Mechanics', 'Industrial and Production Engineering'
The question is:
{question}
"""


cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

# QA generation template for interpreting Cypher results
qa_generation_template = """
You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response in the first person, as if you are Khayrul. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative; you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question **in the first person**.

Query Results:
{context}

Question:
{question}

Guidelines:
- If the provided information is empty (e.g., []), respond with: "I don't have the information to answer that question."
- If the information is not empty, provide a clear and concise answer using the results, speaking in the first person.
- All information pertains to my educational background.
- Never state that you lack information if query results are present.
- Include all relevant query results in your response if applicable.
- B.Sc. or bsc stands for Bachelor of Science.
- M.S. or ms stands for Master of Science or master degree.
- Ph.D. or phd stands for Doctor of Philosophy.

Helpful Answer:
"""



qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

# Initialize the GraphCypherQAChain with the OpenAI model and Neo4j graph
education_chain = GraphCypherQAChain.from_llm(
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
