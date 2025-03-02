from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-o3-mini")

positive_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistent."),
        ("human","generate a thankyou note for this positive feedback:{feedback}.")
    ]
)
negative_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistent."),
        ("human","generate a response addressing this negative feedback:{feedback}.")
    ]
)
nutral_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistent."),
        ("human","generate a request for more details for this nutral feedback:{feedback}.")
    ]
)
escalate_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system","you are an helpful assistent."),
        ("human","generate a message to escalate this human feedback:{feedback}.")
    ]
)

# classify the feedback whether it is {+ve} or {-ve} or {nutral} or escalate..
classification_feedback_template=ChatPromptTemplate.from_messages(
    [
        ("system","your are an helpful assistent"),
        ("user","Classify this sentiment as positive , negative, netural or escalate: {feedback}.")
    ]
)

#Define the runnable branches for feedback handling
branches=RunnableBranch(
    (
        lambda x:"positive" in x,
        positive_feedback_template | llm | StrOutputParser()
    ),
    (
        lambda x:"negative" in x,
        negative_feedback_template | llm | StrOutputParser()
    ),
    (
        lambda x:"nutral" in x,
        nutral_feedback_template | llm | StrOutputParser()
    ),
    escalate_feedback_template | llm | StrOutputParser()
)

#create classification chain
classification_chain= classification_feedback_template | llm | StrOutputParser()

chain= classification_chain | branches

review="it is a terrible product. it is broken in one use it was not good product."

result=chain.invoke({"feedback":review})

print(result)

