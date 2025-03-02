from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(model="gpt-4o")

messages=[
    ("system","you are an expert in generating captions on {topic}."),
    ("human","give me the {caption_count} captions.")
]
prompt_template=ChatPromptTemplate.from_messages(messages)
prompt=prompt_template.invoke({"topic":"AI","caption_count":3})
result=llm.invoke(prompt)
print(result.content)