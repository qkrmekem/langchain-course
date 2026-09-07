from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    return "Tokyo Weather is sunny."

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="What's the weather in Tokyo?")})
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
