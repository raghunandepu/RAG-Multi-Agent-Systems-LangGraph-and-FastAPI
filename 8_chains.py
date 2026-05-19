"""
Understanding Chains in LangChain V.1
LCEL patterns, composition, and debugging
"""

from unittest import result
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0)

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("user", "Summarize the following {text} in one sentence.")
    ])
    parser = StrOutputParser()
    chain = prompt | model | parser
    result = chain.invoke({"text": "LangChain is a powerful framework for building applications with language models."})
    print("Basic Chain Result:", result)

def demo_parallel_chain():
    """Run multiple chains in parallel and combine results."""
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("user", "Summarize the following {text} in one sentence.")
    ])

    keywords_prompt = ChatPromptTemplate.from_messages([
        ("user", "Extract keywords from the following {text}\n .Return a comma-separated list.")
    ])
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("user", "What is the sentiment of the following text? {text}")
    ])

    parser = StrOutputParser()
    parallel_chain = RunnableParallel(
        summarize=summarize_prompt | model | parser,
        keywords=keywords_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
    )

    text = """
    The new AI features are absolutely incredible! Users are loving the
    faster response times and improved accuracy. However, some have noted
    that the pricing could be more competitive. Overall, the product
    launch has been a massive success with record-breaking adoption rates.
    """
    results = parallel_chain.invoke({"text": text})
    print("Parallel Chain Results:", results)
    print("Summary:", results["summarize"])
    print("Keywords:", results["keywords"])
    print("Sentiment:", results["sentiment"])

def demo_passhrough_chain():
    """Demonstrate a chain that includes a passthrough step."""

    prompt = ChatPromptTemplate.from_messages([
        ("user", "What is the capital of France?")
    ])
    passthrough = RunnablePassthrough()
    chain = prompt | model | passthrough
    result = chain.invoke({})
    print("Passthrough Chain Result:", result)

def demo_chain_branching():
    """A chain that demonstrates branching functionality."""

    # Different prompts for different intents
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with: {input}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}"
    )

    # Classifier
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as 'code' or 'general': {input}\nReturn only the classification."
    )
    classifer = classifier_prompt | model | StrOutputParser()

    # Branching chain  based on classification
    def is_code_question(input_dict):
        classification = classifer.invoke(input_dict)
        return "code" in classification.lower()

    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser(),  # default branch
    )

    # Test
    questions = [
        "How do I write a for loop in Python?",
        "What's the weather like today?",
    ]
    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Q: {q}")
        print(f"A: {result[:100]}...\n")


def demo_debbuging():
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | model | StrOutputParser()

    # Method 1: Get configuration
    print("Chain input schema:", chain.input_schema.model_json_schema())
    print("Chain output schema:", chain.output_schema.model_json_schema())

    # Method 2: Use with_config for tacing
    result = chain.with_config(
        run_name="greeting_chain",
        # tags="demo,debugging",
    ).invoke({"name": "Alice"})
    print(f"Greeting: {result}")

    # Method 3: Inspect intermediate steps
    # Using RunnableLambda for logging
    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
        return x

    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "after_prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "after_model"))
        | StrOutputParser()
    )

    print("\nDebug chain execution:")
    result = debug_chain.invoke({"name": "Debug"})
    print(f"Greeting: {result}")


    

if __name__ == "__main__":
    #demo_basic_chain()
    #demo_parallel_chain()
    #demo_passhrough_chain()
    #demo_chain_branching()
    demo_debbuging()