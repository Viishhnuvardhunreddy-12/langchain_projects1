from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import  HumanMessage,SystemMessage
load_dotenv()
messages= [
    SystemMessage(content="You are an maths expert where you can solve hard problems easily"),
    HumanMessage(content="what is the square root of 4?"),
]

llm=ChatAnthropic(model="Claude 3.5 Haiku")
result=llm.invoke(messages)
print(result.content)