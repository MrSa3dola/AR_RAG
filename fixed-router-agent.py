import json
import re
from typing import Any, Dict

from crewai import Agent, Crew, Process, Task

from tools.llm import LLM
from tools.pine_cone_tool import rag
from tools.web_scraper import web_scraper

# Create chat agent for general conversation
chat_agent = Agent(
    llm=LLM.llm(temperature=0.2, max_tokens=500),
    role="Chat Agent",
    goal="Handle general conversation and non-furniture-related queries.",
    backstory="A friendly, knowledgeable assistant for casual conversation and general information.",
    verbose=True,
)

# Create RAG agent for furniture lookup
rag_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="Furniture Knowledge Agent",
    goal="Find the most relevant furniture items using vector search",
    backstory="An expert in interior design with access to a furniture database",
    tools=[rag],
    verbose=True,
)

# Create web scraping agent for dynamic furniture search
scrap_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="Furniture Research Agent",
    goal="Find detailed furniture specifications by searching online marketplaces",
    backstory="A meticulous researcher who can find specific furniture items and their details online",
    tools=[web_scraper],
    verbose=True,
)

# Create the router agent
router_agent = Agent(
    llm=LLM.llm(temperature=0, max_tokens=500),
    role="Query Router",
    goal="Direct each query to the most appropriate specialized agent",
    backstory="""
        An intelligent coordinator that analyzes user queries and delegates to the right specialist.
        Furniture-related queries go to RAG or web scraping agents, while general conversation 
        goes to the chat agent.
    """,
    verbose=True,
)


def format_rag_output(rag_result: Any) -> str:
    """Format the RAG output to be more user-friendly"""
    try:
        if isinstance(rag_result, str):
            # Try to parse if it's a string representation of dict/list
            try:
                data = json.loads(rag_result.replace("'", '"'))
            except:
                # If it's not valid JSON, try to evaluate it as a Python literal
                try:
                    import ast

                    data = ast.literal_eval(rag_result)
                except:
                    # If that fails too, return as is
                    return rag_result
        else:
            data = rag_result

        if isinstance(data, list):
            output = "Here are the furniture items I found for you:\n\n"
            for item in data:
                if isinstance(item, dict):
                    caption = item.get("caption", "Unknown item")
                    price = item.get("price", "Price not available")
                    score = item.get("score", "N/A")
                    output += f"- {caption}\n"
                    if isinstance(price, (int, float)):
                        output += f"  Price: ${price:.2f}\n"
                    else:
                        output += f"  Price: {price}\n"
                    output += f"  Match score: {score}\n\n"
                else:
                    output += f"- {item}\n"
            return output
        elif isinstance(data, dict):
            caption = data.get("caption", "Unknown item")
            price = data.get("price", "Price not available")
            score = data.get("score", "N/A")
            output = f"I found this furniture item for you:\n\n"
            output += f"- {caption}\n"
            if isinstance(price, (int, float)):
                output += f"  Price: ${price:.2f}\n"
            else:
                output += f"  Price: {price}\n"
            output += f"  Match score: {score}\n"
            return output
        else:
            return str(rag_result)
    except:
        # If anything goes wrong, return the original
        return str(rag_result)


def extract_final_answer(scraper_result: str) -> str:
    """Extract the final answer section from scraper output"""
    final_answer_pattern = r"## Final Answer:(.*?)(?:$|#)"
    match = re.search(final_answer_pattern, str(scraper_result), re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return str(scraper_result)


def is_furniture_related(query: str) -> bool:
    """Check if a query is related to furniture"""
    furniture_keywords = [
        "sofa",
        "chair",
        "table",
        "desk",
        "bed",
        "furniture",
        "couch",
        "dresser",
        "cabinet",
        "shelf",
        "bookcase",
        "ottoman",
        "armchair",
        "recliner",
        "loveseat",
        "sectional",
        "nightstand",
        "wardrobe",
    ]
    return any(keyword in query.lower() for keyword in furniture_keywords)


def process_query(query: str) -> str:
    """
    Process a user query through the router system and return the appropriate response.
    """
    # First check if it's furniture related
    if is_furniture_related(query):
        # Check RAG scores first
        rag_check_task = Task(
            description=f"Search for furniture matching: {query}\nReturn raw results with confidence scores.",
            agent=rag_agent,
            expected_output="Raw search results with scores",
        )

        # Get RAG results
        rag_result = rag_agent.execute_task(
            rag_check_task, context={}, inputs={"query": query}
        )

        # Try to parse the result for scores
        try:
            # Look for score patterns in the response
            score_pattern = r'score["\']\s*:\s*([\d\.]+)'
            scores = re.findall(score_pattern, str(rag_result))

            has_high_confidence = False
            if scores:
                scores = [float(s) for s in scores]
                has_high_confidence = any(score >= 0.7 for score in scores)

            if has_high_confidence:
                # Use RAG agent for final result
                rag_final_task = Task(
                    description=f"""
                        Find furniture matching: {query}
                        Format each result with description, price, and confidence score.
                        Only include results with score >= 0.7
                    """,
                    agent=rag_agent,
                    expected_output="Formatted furniture recommendations",
                )
                final_result = rag_agent.execute_task(
                    rag_final_task, context={}, inputs={"query": query}
                )
                return format_rag_output(final_result)
            else:
                # Use Scraper agent
                scraper_task = Task(
                    description=f"""
                        Search online for furniture matching: {query}
                        Find detailed specifications including dimensions, materials, and pricing.
                        Compile the best 2-3 matches with full details and direct links.
                        
                        ## Final Answer:
                        [Full product descriptions with links]
                    """,
                    agent=scrap_agent,
                    expected_output="Detailed furniture listings with specifications",
                )
                final_result = scrap_agent.execute_task(
                    scraper_task, context={}, inputs={"query": query}
                )
                return extract_final_answer(final_result)

        except Exception as e:
            # Fallback to scraper if parsing fails
            scraper_task = Task(
                description=f"""
                    Search online for furniture matching: {query}
                    Find detailed specifications including dimensions, materials, and pricing.
                    Compile the best 2-3 matches with full details and direct links.
                    
                    ## Final Answer:
                    [Full product descriptions with links]
                """,
                agent=scrap_agent,
                expected_output="Detailed furniture listings with specifications",
            )
            final_result = scrap_agent.execute_task(
                scraper_task, context={}, inputs={"query": query}
            )
            return extract_final_answer(final_result)
    else:
        # Use chat agent for non-furniture queries
        chat_task = Task(
            description=f"Engage in a friendly conversation to answer: {query}",
            agent=chat_agent,
            expected_output="A conversational, informative response",
        )
        final_result = chat_agent.execute_task(
            chat_task, context={}, inputs={"query": query}
        )
        return str(final_result)


# Main entry point for handling queries
def handle_query(query: str) -> str:
    """Main handler function for user queries"""
    return process_query(query)


# Example usage
if __name__ == "__main__":
    # Test with example query
    query = "I'm looking for a light blue sofa that would fit in my small apartment"
    result = handle_query(query)
    print("Final Result:")
    print(result)
