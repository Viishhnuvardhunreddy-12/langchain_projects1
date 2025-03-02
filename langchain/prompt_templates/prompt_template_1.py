from langchain_openai import ChatOpenAI
from dotenv import load_dotnev
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
llm=ChatOpenAI(model="gpt=3.5-turbo")
template="Write a {tone} Email to {company} expressing intrest in {position} position,mentioning {skill} as key strength.keep it in 4 lines max."
prompt_template=ChatPromptTemplate.from_messages(template)
prompt=prompt_template.invoke({
    "tone":"Professional",
    "company":"Google",
    "position":"AI Engineer",
    "skill":"AI"
})
result=llm.invoke(prompt)
print(result.content)