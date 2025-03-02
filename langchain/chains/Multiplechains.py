from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-3.5-turbo")
prompt_template=ChatPromptTemplate.from_messages([
    ("system","you are a best explorer of world tell me facts on {country}"),
    ("user","tell me the {count} facts"),
])

# it is used to translate the above output into the targeted language
translation_template=ChatPromptTemplate.from_messages([
    ("system","you are an language translator convert the provided text into any language given by the user."),
    ("user","Translate the following text into {language}"),
])
prepare_translation=RunnableLambda(lambda output: {"text":output,"language":"french"})

chain= prompt_template | llm | StrOutputParser() | prepare_translation | translation_template | llm | StrOutputParser()

result=llm.invoke({"country":"india","count":2})
print(result)