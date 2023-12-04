import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

MODEL_NAME = "gpt-3.5-turbo"

import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, get_all_tool_names, AgentType
from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.memory import ConversationBufferMemory

tool_names = ['news-api', 'python_repl', 'requests_all', 'requests_get', 'requests_post', 
              'requests_patch', 'requests_put', 'requests_delete', 'wolfram-alpha', 
              'google-search', 'google-search-results-json', 'dalle-image-generator']

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = load_tools(tool_names=tool_names, llm=llm, news_api_key=NEWS_API_KEY)

agent = initialize_agent(tools=tools,
                         llm=llm,
                         memory=memory,
                         agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                         verbose=True, 
                         handle_parsing_errors=True)

@cl.on_message
async def main(message: cl.Message):
    response = ""
    elements = []
    try:
        response = agent.run(message.content)
        image_url = DallEAPIWrapper().run(response)
        if image_url:
            elements.append(
                cl.Image(name="generated_image", display="inline", url=image_url)
            )
    except Exception as e:
        response = f"Sorry, I have an internal error: {e}"
        
    await cl.Message(
        content=response,
        elements=elements
    ).send()

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Hello there!  I am Sydney.  How are you?"
    ).send()
