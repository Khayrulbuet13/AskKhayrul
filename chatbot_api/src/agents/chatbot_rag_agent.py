import os
from chains.education_chain import education_chain
from chains.papers_chain import papers_chain
from tools.papersQA import papers_qa_chain
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI


AGENT_MODEL = os.getenv("AGENT_MODEL")

chatbot_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

# Define the wrapper function
def papers_qa_tool_func(input_text):
    output = papers_qa_chain({'query': input_text})
    return output['result']

# tools = [
    

#     Tool(
#         name="EducationGraph",
#         func=education_chain.invoke,  # Update the function name according to your chain
#         description="""Useful for answering questions about Khayrul's education background, 
#         including institutions, degrees, GPA, thesis titles, and courses taken. Use the entire 
#         prompt as input to the tool. For instance, if the prompt is "What is Khayrul's GPA 
#         for his Ph.D.?", the input should be "What is Khayrul's GPA for his Ph.D.?"
#         """,
#     ),
#     Tool(
#         name="PaperGraph",
#         func=papers_chain.invoke,  # Ensure this points to your papers_chain
#         description="""Useful for answering questions about Khayrul's published papers and the 
#         skills he has utilized or gained through them. Use the entire prompt as input to the tool.
#         For instance,If the prompt is "What softwares are utilized in Khayrul's latest paper?",
#         the input should be "What softwares are utilized in Khayrul's latest paper?".
#         """
#     ),

#     Tool(
#     name="PapersQA",
#     func=papers_qa_tool_func,
#     description="""Useful for answering explicit details about Khayrul's published papers.
#     Use this tool when the user asks detailed questions about the content, methodology,
#     results, or other specifics of the papers. Input should be the user's question.""",
# )

# ]


tools = [
    # Tool(
    #     name="EducationGraph",
    #     func=education_chain.invoke,
    #     description="""Use this tool to answer questions about your education background, 
    #     including institutions, degrees, GPA, thesis titles, and courses taken. 
    #     Respond in the first person.""",
    # ),

    Tool(
        name="EducationGraph",
        func=education_chain.invoke,
        description="""Use this tool to answer questions about your education background, 
        including institutions, degrees, GPA, thesis titles, and courses taken. 
        Always provide the entire question as input to this tool. For example, 
        if asked "What courses are included in the Ph.D. program?", 
        pass the entire question to the tool.""",
    ),

    Tool(
        name="PaperGraph",
        func=papers_chain.invoke,
        description="""Use this tool to answer questions about your published papers and 
        the skills you've utilized or gained through them. Always provide the entire 
        question as input to this tool. For example, if asked 
        "What are the papers published by you as the first author?", 
        pass the entire question to the tool.""",
    ),
    Tool(
        name="PapersQA",
        func=papers_qa_tool_func,
        description="""Use this tool to answer detailed questions about the content,
        methodology, results, or specifics of your published papers. 
        Always provide the entire question as input to this tool. For example, if asked 
        "Could you explain the main concept and significance of your paper titled 'Multiplex Image Machine Learning'?", 
        pass the entire question to the tool.""",
    ),
]

chat_model = ChatOpenAI(
    model=AGENT_MODEL,
    temperature=0,
)

chatbot_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=chatbot_agent_prompt,
    tools=tools,
)

chatbot_rag_agent_executor = AgentExecutor(
    agent=chatbot_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)
