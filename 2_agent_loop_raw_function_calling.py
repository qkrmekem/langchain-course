from dotenv import load_dotenv

load_dotenv()

import ollama

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"    >> Executing get_product_price(product=`{product}`)")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# 차이점 2: @tool을 안 쓰면 함수마다 JSON 스키마를 직접 다 써줘야 한다.
# LangChain의 @tool 데코레이터는 함수의 타입 힌트와 docstring을 보고
# 아래 내용을 알아서 만들어 주던 부분이다.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# 참고: Ollama도 함수를 그대로 tools에 넘기면 이 스키마를 알아서 만들어 준다.
# (LangChain의 @tool 데코레이터와 비슷하다)
#   tools_for_llm = [get_product_price, apply_discount]
# 다만 이러려면 docstring을 Google 스타일로 써야 한다. 그래야 Ollama가
# Args 섹션에서 파라미터 설명을 읽어갈 수 있다. 예를 들면 이런 식이다:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# 여기서는 @tool이 가려 주던 게 뭔지 눈으로 보려고 JSON을 직접 쓰는 방식을 남겨 뒀다.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# --- Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    # 툴 모음
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # 초기 메시지 설정
    messages = [
        {
            "role": "system",
            "content": (
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
        },
        {"role": "user", "content": question},
    ]

    # 에이전트 루프 시작
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # 차이점 5: llm_with_tools.invoke() 대신 ollama.chat()을 그대로 호출한다.
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        # 호출된 툴 확인
        tool_calls = ai_message.tool_calls

        # 툴 호출이 없으면 최종 답변으로 간주
        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # 첫번째 툴만 획득 - 한번에 하나씩만 실행하도록
        tool_call = tool_calls[0]
        # 차이점 6: dict처럼 .get("name")으로 꺼내는 게 아니라 .function.name 속성으로 접근한다.
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"    [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None: 
            raise ValueError(f"Tool '{tool_name}' not found.")

        # 차이점 7: tool.invoke()가 아니라 그냥 파이썬 함수를 직접 호출한다.
        # 툴 호출 및 결과 관찰
        observation = tool_to_use(**tool_args)

        print(f"    [Tool Result] {observation}")

        # AI 메시지와 툴 결과를 메시지 히스토리에 추가
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")