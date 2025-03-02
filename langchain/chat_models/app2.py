from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

chat_history=[]
system_message=SystemMessage(content="you are the helpful assistant")
chat_history.append(system_message)

llm=ChatOpenAI(model="gpt-4.0")

while True:
    query=input("you: ")
    if query.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=query))
    result=llm.invoke(chat_history)
    response=result.content
    chat_history.append(AIMessage(content=response))
    print("AI: {response}")

print("Message History")
print(chat_history)