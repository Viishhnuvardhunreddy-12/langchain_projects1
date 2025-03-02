from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence,RunnableLambda
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-4omini")
prompt_template=ChatPromptTemplate.from_messages([
    ("system","you love to explare facts tell me that fact about {animal}"),
    ("user","tell me {fact_count} facts"),
])
format_prompt=RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_llm=RunnableLambda(lambda x:llm.invoke(x.to_messages()))
parse_output=RunnableLambda(lambda x: x.content)

chain=RunnableSequence(first=format_prompt,middle=[invoke_llm],last=parse_output)
result=llm.invoke({"animal":"cat","fact_count":1})
print(result)           # Instead of this it is better to use | pipe operator as in the chain_1.py