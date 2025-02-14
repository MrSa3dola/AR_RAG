from crewai import Agent, Crew, Process, Task

from tools.llm import LLM

chat_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=2000),
    role="chat_agent",
    goal="Handle general conversation and non-furniture-related queries.",
    backstory="A friendly, knowledgeable chat agent for casual conversation.",
    # allow_delegation=True,
    verbose=True,
)

chat_task = Task(
    description="""Act as a general chatbot to answer {query}""",
    agent=chat_agent,
    expected_output="A friendly conversation response.",
)

crew = Crew(
    manager_llm=LLM.llm(temperature=0),
    agents=[chat_agent],
    tasks=[chat_task],
    # memory=True,
    # process=Process.hierarchical,
)
query = "hello"
inputs = {"query": query}
result = crew.kickoff(inputs=inputs)
print(result)
