import asyncio
import json
import os
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


with open("orders.json", "r", encoding="utf-8") as f:
    ORDERS = json.load(f)


def lookup_order(order_id: str) -> dict[str, Any]:
    return ORDERS.get(order_id, {"error": f"Order {order_id} not found"})


def calculate(expression: str) -> float:
    try:
        return float(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return {"error": f"Could not evaluate expression: {exc}"}


def build_tools():
    return [
        lambda order_id: lookup_order(order_id),
        lambda expression: calculate(expression),
    ]


# The ADK FunctionTool expects a callable, so we wrap each helper directly.
def lookup_order_tool(order_id: str) -> dict[str, Any]:
    print(f"[tool] lookup_order(order_id={order_id})")
    result = lookup_order(order_id)
    print(f"[tool] result: {result}")
    return result


def calculate_tool(expression: str) -> float | dict[str, str]:
    print(f"[tool] calculate(expression={expression})")
    result = calculate(expression)
    print(f"[tool] result: {result}")
    return result


def create_agent() -> LlmAgent:
    return LlmAgent(
        name="orders_assistant",
        model="gemini-2.5-flash",
        description="A helpful assistant that looks up order details and does simple arithmetic.",
        instruction=(
            "You are a helpful orders assistant. Use the lookup_order tool to inspect order details "
            "from the catalog when the user mentions an order ID. Use the calculate tool for arithmetic. "
            "If an order is not found, say clearly that the order could not be found and do not invent details."
        ),
        tools=[lookup_order_tool, calculate_tool],
        generate_content_config=types.GenerateContentConfig(temperature=0),
    )


async def run_goal_async(goal: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    os.environ["GOOGLE_API_KEY"] = api_key

    agent = create_agent()
    session_service = InMemorySessionService()
    app_name = "orders_app"
    session = session_service.create_session_sync(app_name=app_name, user_id="student")
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    print(f"[goal] {goal}")
    print("[trace] starting agent run")

    final_text = []
    async for event in runner.run_async(
        user_id="student",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=goal)]),
    ):
        if getattr(event, "content", None):
            content = event.content
            parts = getattr(content, "parts", []) or []
            text_bits = [getattr(part, "text", "") for part in parts if getattr(part, "text", None)]
            if text_bits:
                text = "\n".join(text_bits)
                print(f"[trace] {text}")
                if event.is_final_response():
                    final_text.append(text)
            elif getattr(event, "actions", None):
                print(f"[trace] event actions: {event.actions}")

    return "\n".join(final_text).strip() or "No final answer produced."


def run_goal(goal: str) -> str:
    return asyncio.run(run_goal_async(goal))


if __name__ == "__main__":
    goal = "I'm thinking of buying two more of order A1001. What would those two cost, and is the original still under warranty?"
    answer = run_goal(goal)
    print("[final]", answer)
