"""
Working with LLMs in Langchain v1
Multiple providers, configuration, streaming and cost optimization
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def demo_init_chat_model():
    """Demonstrates initializing a chat model using langchain_core's init_chat_model"""
    llm = init_chat_model(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        streaming=True,
        max_retries=3,
    )
    response = llm.invoke("Say 'Hello from init_chat_model!' in one word")
    print(f"Response from init_chat_model: {response}")


def demo_model_comparison():
    prompt = "Explain recursion in one sentence."

    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False,
        ),
        "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=False,
        ),
    }

    # add anthropic model if available
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5-20250929"] = init_chat_model(
            model="claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=0.7,
            streaming=False,
        )

    print(f"Prompt: {prompt}\n")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")


def demo_message():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]
    # print("Using message objects:")
    # print(f"Messages: {messages[0]} | {messages[1]}")

    response = model.invoke(messages)
    print(f"Response from the Pirate: {response.content}")

    # Multi-turn conversation using message objects
    messages.append(response)  # add model's response to the conversation
    messages.append(HumanMessage(content="What about tomorrow?"))

    print("\nMulti-turn conversation:")
    response = model.invoke(messages)
    print(f"Follow-up response from the Pirate: {response.content}")

def exercise_multi_model(question: str, models: list[str]) -> dict[str, str]:
    """
    EXERCISE: Create a function that:
    1. Takes a question and a list of model names
    2. Gets responses from all models
    3. Returns a dict of {model_name: response}

    Test with: question="What is AI?", models=["gpt-4o-mini", "gpt-4o"]
    """
    responses = {}
    for model_name in models:
        model = init_chat_model(model=model_name, temperature=0.7, streaming=False)
        response = model.invoke(question)
        responses[model_name] = response.content

    return responses


if __name__ == "__main__":
    #demo_init_chat_model()
    #demo_model_comparison()
    #demo_message()
    results = exercise_multi_model("What is AI?", ["gpt-4o-mini", "gpt-4o"])
    print("Responses from multiple models:")
    for model_name, response in results.items():
        print(f"{model_name}: {response}")
