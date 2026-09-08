from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

# llm 모델과 마찬가지로 초기화 시점에 환경변수에서 정해진 규약에 따라 api 키를 읽어와 초기화
# tavily = TavilyClient()

# 검색 툴 정의(사용자 정의)
# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query = query)

llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=32768)
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
