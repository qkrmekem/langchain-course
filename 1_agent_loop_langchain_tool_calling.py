from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"    >> Executing get_product_price(product=`{product}`)")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# --- Agent Loop ---

@traceable(name="Langchain Agent Loog")
def run_agent(question: str):
    # 툴 모음
    tools = [get_product_price, apply_discount]
    # 툴 이름을 키로 하는 딕셔너리 생성
    tools_dict = {t.name: t for t in tools}

    # 챗 모델 초기화
    llm = init_chat_model(f"ollama:{MODEL}", tamperature=0)
    # 챗 모델에 툴 바인딩
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    # 초기 메시지 설정
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product prices."
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # 에이전트 루프 시작
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # AI 메세지 생성
        ai_message = llm_with_tools.invoke(messages)

        # 호출된 툴 확인
        tool_calls = ai_message.tool_calls

        # 툴 호출이 없으면 최종 답변으로 간주
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # 첫번째 툴만 획득 - 한번에 하나씩만 실행하도록
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None: 
            raise ValueError(f"Tool '{tool_name}' not found.")

        # 툴 호출 및 결과 관찰
        observation = tool_to_use.invoke(tool_args)

        print(f"    [Tool Result] {observation}")

        # AI 메시지와 툴 결과를 메시지 히스토리에 추가
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id= tool_call_id)
        )

    print("Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")