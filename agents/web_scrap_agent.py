from crewai import Agent, Crew, Task

from tools.llm import LLM
from tools.web_scraper import web_scraper

scrap_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=1000),
    role="Furniture Recommendation Agent",
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
    verbose=False,
)
scraper_task = Task(
    description=(
        f"including dimensions, design details, material composition, and functionality. Compare these "
        "details against the following input criteria: {query} "
        "and identify the top  best fit the description."
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

crew = Crew(agents=[scrap_agent], tasks=[scraper_task])
result = crew.kickoff(inputs={"query": "light blue chaise longue"})
print(result)
