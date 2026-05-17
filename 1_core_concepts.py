from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model


load_dotenv()

def demo_basic_chain():
    """Demonstrates a basic chain using LCEL and runnables"""
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("You are a helpful assistant. Answer in one sentence: {question}")
    ])

    # Create an output parser
    output_parser = StrOutputParser()

    # Create a ChatOpenAI instance
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Compose with pipe operator
    chain = prompt | llm | output_parser
    #parsed_response = chain.invoke({"question": "What is the capital of USA?"})
    #print(f"Response from ChatOpenAI: {parsed_response}")

    return chain

def demo_batch_execution():
    """Demonstrates batch execution for multiple inputs"""
    chain = demo_basic_chain()
    batch_inputs = [
        {"question": "What is the capital of USA?"},
        {"question": "What is the capital of Germany?"},
        {"question": "What is the capital of Japan?"}
    ]
    batch_responses = chain.batch(batch_inputs)
    for i, response in enumerate(batch_responses):
        print(f"Batch {i + 1} Response: {response}")

def demo_streaming():
    """Demonstrates streaming responses from the LLM"""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    parser = StrOutputParser()
    chain = prompt | llm | parser

    response_stream = chain.stream({"topic": "nature"})
    print("Streaming response:")
    for chunk in response_stream:
        print(chunk, end="", flush=True)
    print()  # for newline after streaming is done

def demo_schema_inspection():
    """Demonstrates input/output schema inspection for a chain"""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    chain = prompt | llm | parser

    print("Input schema:")
    print(chain.input_schema.model_json_schema())

    print("Output schema:")
    print(chain.output_schema.model_json_schema())

def exercise_first_chain():
      """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """
      prompt = ChatPromptTemplate.from_template("Create a marketing tagline for a product called " \
      "'{product}' targeting '{audience}'.")
      llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
      parser = StrOutputParser()
      chain = prompt | llm | parser
      response = chain.invoke({"product": "AI Course", "audience": "developers"})
      print(f"Generated tagline: {response}")
      return response      


def newWayToInitilaize():
     # the univeral way to initialize a model using init_chat_model
     model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)
     response = model.invoke("What is the meaning of life?")
     print(f"Response from model: {response}")



if __name__ == "__main__":
    #demo_batch_execution()
    #demo_streaming()
    #demo_streaming()
    #demo_schema_inspection()
    exercise_first_chain()