from crewai import Agent, Crew, Task

from tools.llm import LLM

chat_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=2000),
    role="chat_agent",
    goal="Handle general conversation and non-furniture-related queries.",
    backstory="A friendly, knowledgeable chat agent for casual conversation.",
    allow_delegation=True,
    verbose=True,
)

chat_task = Task(
    description="Chat with the user about: {query}",
    agent=chat_agent,
    expected_output="A friendly conversation response.",
)
