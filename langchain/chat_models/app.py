from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model='gpt-3.5')
result=llm.invoke("what are the multiples of 9")
print(result.content)