from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-3.5-turbo")
prompt_template=ChatPromptTemplate.from_message(
    [
        ('system','you are an expert who knows facts on {topic}.'),
        ('human','tell em the {fact_count} facts on it.')
    ]
)
chain= prompt_template | llm | StrOutputParser()
result=chain.invoke({"topic":"planets","fact_count":2})
print(result)  # no need to write result.content becoz of StrOutputParser() it can handle output well...