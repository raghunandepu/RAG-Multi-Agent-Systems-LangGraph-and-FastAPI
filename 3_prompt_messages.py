"""
This file demonstrates how to use different types of prompt templates in LangChain, including 
ChatPromptTemplate and FewShotChatMessagePromptTemplate. It also shows how to format prompts with 
variables and how to use these prompts with a chat model."""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ChatMessage, ToolMessage
from langchain_core.prompts import FewShotChatMessagePromptTemplate

load_dotenv()

#chatprompttemplate example
def demo_chat_prompt_template():
    """Demonstrates using ChatPromptTemplate to create a prompt for a chat model"""
    # prompt = ChatPromptTemplate.from_template(
    #     "You are a helpful assistant that explains concepts in simple terms. {question}"
    # )
    # formatted_prompt = prompt.format_messages(question="What is recursion?")
    # print("Formatted Prompt:")
    # for message in formatted_prompt:
    #     print(f"{message.type}: {message.content}")

    
    #multi-message template example
    multi_message_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
            ("human", "Translate the following text: {text}"),
        ]
    )
    formatted_multi_message_prompt = multi_message_prompt.format_messages(
        input_language="English",
        output_language="Hindi",
        text="Hello, how are you?"
    )
    print("\nFormatted Multi-Message Prompt:")
    for message in formatted_multi_message_prompt:
        print(f"{message.type}: {message.content}")
    
    model = init_chat_model(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=False,
    )
    response = model.invoke(formatted_multi_message_prompt)
    print(f"\nResponse from model: {response.content}")


#Fewshot example
def demo_fewshot_chat_message_prompt_template():
    examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    ]

    example_prompt = ChatPromptTemplate.from_messages({
    ("human", "{input}"),
    ("ai", "{output}"),
    })

    fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Give the opposite of each word."),
            fewshot_prompt,
            ("human", "{input}"),
        ]
    )


    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    response = model.invoke(final_prompt.format_messages(input="happy"))

    print(response.content)



if __name__ == "__main__":
    #demo_chat_prompt_template()
    demo_fewshot_chat_message_prompt_template()