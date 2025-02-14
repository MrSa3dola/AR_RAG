from crewai import Agent, Crew, Process, Task
from langchain_openai import ChatOpenAI

from tools.llm import LLM
from tools.pine_cone_tool import rag
from tools.web_scraper import web_scraper

OpenAIGPT4 = ChatOpenAI(model="gpt-4")

chat_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="chat_agent",
    goal="Handle general conversation and non-furniture-related queries.",
    backstory="A friendly, knowledgeable chat agent for casual conversation.",
    allow_delegation=False,
    Memory=False,
    # max_iter=1,
    verbose=True,
)

chat_task = Task(
    description="""Act as a general chatbot to answer {query}""",
    agent=chat_agent,
    expected_output="A friendly conversation response.",
)

scrap_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="scrap_agent",
    goal=(
        "analyze furniture_listings from amazon of egypt, comparing each product's "
        "specifications against user-defined input criteria to recommend the best match."
    ),
    backstory=(
        "Leveraging advanced web serch techniques and natural language processing, "
        "this agent navigates the amazon platform to extract detailed product specifications "
        "from furniture_listings. It evaluates attributes such as design, dimensions, materials, "
        "and functionality to provide a tailored recommendation based on the provided criteria."
    ),
    tools=[web_scraper],
    # context=[rag_task],
    allow_delegation=False,
    # max_iter=1,
    verbose=True,
)

scraper_task = Task(
    description=(
        "including dimensions, design details, material composition, and functionality. Compare these "
        "details against the following input criteria: {query} "
        "and identify the top best fit the description."
        "if you don't found any product so say no furniture exist."
        "just show after the( ## Final Answer:) "
        "write just the results"
        "give me the url of each product that you give me in the description"
    ),
    agent=scrap_agent,
    expected_output=(
        "the best 2 fit furniture with the input description"
        "Each one in desciption format"
        "in string format not json format"
        "just show after the( ## Final Answer:) "
        "write just the results"
    ),
)

rag_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="rag_agent",
    goal="Use Pinecone to find furniture matching the user's query",
    backstory="An expert in interior design and furniture recommendations",
    tools=[rag],
    # context=[chat_task],
    subordinates=[scrap_agent],
    allow_delegation=False,
    # max_iter=1,
    verbose=True,
)

rag_task = Task(
    description="""Find furniture matching: {query}""",
    agent=rag_agent,
    expected_output="A list of furniture items",
)


router_agent = Agent(
    # llm=LLM.llm(temperature=0, max_tokens=500),
    llm=OpenAIGPT4,
    role="Router",
    goal="""
        Your task is to analyze an incoming {query} and choose the single best specialized agent to delegate it to. 
        You must select exactly one agent for each query, with no subsequent fallbacks or additional processing.

        You have three experts in your system:
        - The rag_agent: excels at high-confidence vector retrieval of furniture recommendations using Pinecone. It returns a score for each recommendation.
        - The scrap_agent: specializes in dynamically extracting detailed product listings and specifications from online sources.
        - The chat_agent: handles all other conversational or non-furniture queries.

        Follow this strict decision process:
        1. Evaluate if the query is furniture-related or not.
        2. For furniture-related queries:
        - First check with the rag_agent
        - If the rag_agent returns at least one result with a confidence score above 0.9, select ONLY the rag_agent and return its results.
        - If all rag_agent results have scores below 0.9 or no results are found, select ONLY the scrap_agent and return its results.
        3. For non-furniture-related or purely conversational queries:
        - Select ONLY the chat_agent and return its response.

        Important: Once you've selected an agent, do not attempt to use any other agents for the same query. Each query must be handled by exactly one agent, with no additional processing afterward.
    """,
    backstory="""
        You are the Manager Router, a highly perceptive and experienced decision-maker trained on vast amounts of specialized and general language data. 
        With deep domain expertise in natural language understanding and retrieval-augmented systems, you excel at discerning subtle contextual clues.
        Your critical role is to ensure every user query is handled by the expert best suited to provide an accurate, timely, and satisfying response.
        Your decisions improve the overall user experience by directing queries to the most capable agent, saving time and boosting precision.
    """,
    verbose=True,
    allow_delegation=True,
    # max_iter=1,
    # allowed_agents=["chat_agent", "rag_agent", "scrap_agent"],
)

router_task = Task(
    description="""
        When presented with a {query}, first determine its domain and intent.
        Ask yourself:
          - Is this query specifically about furniture or interior design?
          - If so, simulate running the rag_agent:
              • If the rag_agent returns one or more results with high confidence above 0.9,
                choose "rag_agent".
              • Otherwise, if the rag_agent's scores are low or no high-confidence results are found, choose "scrap_agent".
          - If the query is not furniture-related or is purely conversational,
            choose "chat_agent".
        Return the final answer from the selected Agent ONLY and stop the process.
    """,
    agent=router_agent,
    expected_output="""
    one of the following outputs
    - rag agent: list of furniture items
    - chat agent: casual conversation
    - web scrap agent: list of items urls
    """,
)


crew = Crew(
    # manager_llm=LLM.llm(temperature=0),
    # manager_llm=OpenAIGPT4,
    manager_agent=router_agent,
    agents=[
        chat_agent,
        rag_agent,
        scrap_agent,
    ],
    tasks=[router_task, chat_task, rag_task, scraper_task],
    # memory=True,
    process=Process.hierarchical,
)
query = "light blue sofa"
inputs = {"query": query}
result = crew.kickoff(inputs=inputs)
print(result)
# while True:

# Your response must be a single string naming the best agent from the list: 'Furniture RAG Agent', 'Web Scraping Agent', or 'General Chat Agent'. Do not include any extra commentary—simply return the chosen agent’s name.
# Return a single string naming the chosen agent: either "rag_agent", "scrap_agent", or "chat_agent".
