import logging
import os, requests
from neo4j import GraphDatabase
from retry import retry

# Environment variables
EDUCATION_CSV_PATH = os.getenv("EDUCATION_CSV_PATH")
COURSES_CSV_PATH = os.getenv("COURSES_CSV_PATH")
PAPERS_JSON_PATH = os.getenv("PAPERS_JSON_PATH")
SKILLS_CSV_PATH = os.getenv("SKILLS_CSV_PATH")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)

NODES = ["Education", "Course", "Skill", "Paper"]

def set_uniqueness_constraints(session, node):
    """Creates a uniqueness constraint for a node on the 'id' property."""
    query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node}) REQUIRE n.id IS UNIQUE"
    session.run(query)
    LOGGER.info(f"Uniqueness constraint ensured for {node} on id.")

@retry(tries=100, delay=10)
def load_educational_data():
    """Loads educational data into Neo4j from specified CSV paths."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session(database="neo4j") as session:
        # Setting uniqueness constraints on all nodes
        for node in NODES:
            set_uniqueness_constraints(session, node)

        # Loading data from education CSV
        query_education = f"""
            LOAD CSV WITH HEADERS FROM '{EDUCATION_CSV_PATH}' AS row
            MERGE (e:Education {{
                id: row.Education_ID,
                institution: row.Institution,
                location: row.Location,
                degree: row.Degree,
                major: COALESCE(row.Major, 'Not Specified'),
                gpa: COALESCE(row.GPA, 'Not Available'),
                startDate: COALESCE(row.StartDate, 'Date Not Provided'),
                endDate: COALESCE(row.EndDate, 'Date Not Provided'),
                awards: COALESCE(row.Awards, 'None'),
                thesisTitle: COALESCE(row.ThesisTitle, 'Not Provided')
            }});
        """
        session.run(query_education)
        LOGGER.info("Education data loaded successfully.")

        # Loading data from courses CSV
        query_courses = f"""
            LOAD CSV WITH HEADERS FROM '{COURSES_CSV_PATH}' AS row
            MERGE (c:Course {{
                id: row.ID,
                EducationID: row.Education_ID,
                courseID: row.CourseID,
                courseName: row.CourseName,
                courseGrade: row.CourseGrade
            }});
        """
        session.run(query_courses)
        LOGGER.info("Course data loaded successfully.")

        # Loading data from skills CSV
        query_skills = f"""
            LOAD CSV WITH HEADERS FROM '{SKILLS_CSV_PATH}' AS row
            MERGE (s:Skill {{
                id: toInteger(row.ID),
                skill: row.skill,
                skill_type: row.skill_type
            }})
        """
        
        session.run(query_skills)
        LOGGER.info("Skill data loaded.")

        
        # Query to merge paper data into the Neo4j database

        # Fetching JSON data from URL
        response = requests.get(os.getenv("PAPERS_JSON_PATH"))
        papers = response.json()

        # Adjust the paper dictionary preparation step
        for paper in papers:
            paper['abstract_challenge'] = paper['abstract'].get('challenge', '')
            paper['abstract_novelty'] = paper['abstract'].get('novelty', '')
            paper['abstract_result'] = paper['abstract'].get('result', '')

        query_papers = """
        UNWIND $papers AS paper
        MERGE (p:Paper {
            id: paper.id,
            title: paper.title,
            author: paper.author,
            first_author: paper.first_author,
            journaltitle: paper.journaltitle,
            publisher: paper.publisher,
            volume: paper.volume,
            issue: paper.issue,
            pages: paper.pages,
            date: paper.date,
            doi: paper.doi,
            url: paper.url,
            keywords: paper.keywords,
            skills: paper.skills,
            abstract_challenge: paper.abstract.challenge,
            abstract_novelty: paper.abstract.novelty,
            abstract_result: paper.abstract.result,
            contribution: paper.contribution
        })
        """

        
        session.run(query_papers, {'papers': papers})
        LOGGER.info("Paper data loaded successfully.")


        # Creating relationships between Education and Courses
        query_relationships = f"""
            LOAD CSV WITH HEADERS FROM '{COURSES_CSV_PATH}' AS courses
                MATCH (e:Education {{id: courses.Education_ID}})
                MATCH (c:Course {{id: courses.ID}})
                MERGE (e)-[:INCLUDED_IN]->(c)
            """
        session.run(query_relationships)
        LOGGER.info("Relationships between Education and Courses established.")

        # Creating relationships between Paper and Skills
        query_paper_skill_relationships = """
        MATCH (p:Paper)
        WITH p, SPLIT(p.skills, ",") AS skillIds
        UNWIND skillIds AS skillId
        WITH p, TOINTEGER(skillId) AS intSkillId
        MATCH (s:Skill {id: intSkillId})  // Only match the skills relevant to the current paper
        MERGE (p)-[:UTILIZES]->(s)
        """

        session.run(query_paper_skill_relationships)
        LOGGER.info("Relationships between Paper and Skills established.")



if __name__ == "__main__":
    load_educational_data()
