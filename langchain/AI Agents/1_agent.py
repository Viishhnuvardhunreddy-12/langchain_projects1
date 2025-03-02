from langchain_openai import ChatOpenAI
from lanchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub  #used for accessing the promps of others from the internet
from langchain.agents import create_react_agent,AgentExecutor
from dotenv import load_dotenv
import datetime
from langchain.agents import tool
load_dotenv()

@tool
def get_system_time(format:str="%Y-%M-%D %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    current_time=datetime.datetime.now()
    formatted_time=current_time.strftime(format)
    return formatted_time

llm=ChatGoogleGenertiveAI(model="gemini-1.5-pro")
#query

query="what is the current time?"  # "what is the weather in paris + 5"

#prompt_template=PromptTemplate.from_messages("{input}") as this is an agent we are pulling prompt from langchainwebsite
prompt_template=hub.pull("hwchase17/react")

tools=[get_system_time]

agent=create_react_agent(llm,tools,prompt_template)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

result=agent_executor.invoke({"input":query})

print(result)