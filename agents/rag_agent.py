from crewai import Agent, Crew, Task

from tools.llm import LLM
from tools.pine_cone_tool import rag

rag_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=1000),
    role="Furniture Recommender",
    goal="Use Pinecone to find furniture matching the user's query",
    backstory="An expert in interior design and furniture recommendations",
    tools=[rag],
    verbose=True,
)

rag_task = Task(
    description="Find furniture matching: {query}",
    agent=rag_agent,
    expected_output="A list of furniture items",
)

crew = Crew(agents=[rag_agent], tasks=[rag_task])
result = crew.kickoff(inputs={"query": "green chaise longue"})
