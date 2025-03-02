from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

current_dir=os.path.dirname(os.path.abspath(__file__))
persistant_directory=os.path.join(current_dir,"db","chroma_db")
#embeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-small")
#load the existing vector store with embedding function
db=Chroma(persistant_directory=persistant_directory,embedding_functions=embeddings)

#user query
query="#insert any query related to the pdf"

#to retrive the relevent document based on the query
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3,"score_threshold":0.5},
)
relevent_docs=retriever.invoke(query)

#print the relevent data
for i,doc in enumarate(relevent_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metedata:
        print(f"Source: {doc.metadata.get('source','Unknown')}\n")