from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

# BaseModel: pydantic의 기반 클래스
# - 검증 + 변환 + JSON Schema 생성 기능을 제공
# Field: pydantic에서 제공하는 필드 정의 클래스 - 필드 단위에서 검증, 변환, 메타데이터 정의 가능
class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent reponse with answer and sources"""

    answer:str = Field(description="The agents's answer to the query")
    source: List[Source] = Field(default_factory=list, description="List of sourced used to generate the answer")

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

# 로컬 모델(qwen2.5:7b)은 구조화 출력 툴을 호출하지 못해 structured_response가 None으로 남음
# - model.profile에 structured_output이 없어 ToolStrategy로 동작하기 때문
# llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=32768)
llm = ChatOpenAI(model="gpt-5-mini")
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
    system_prompt=(
        "You are a job search assistant.\n"
        "Use the tavily_search tool to gather information. "
        "If the results are not enough, search again with a different query."
    ),
)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
            )
        }
    )
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
