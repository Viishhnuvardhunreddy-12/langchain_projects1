import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

#path setting (first we have to add pdfs into the documents folder) documents.
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path=os.path.join(current_dir,"documents","lords_of_rings")
persistant_directory=os.path.join(current_dir,"db","chroma_db")

#check whether chroma vector store already exists 
if not os.path.exists(persistant_directory):
    print("persistant directory doesnot exists. initializing the vextor store.")
    
    #Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exists. please check the path"
        )
        
    #read the text content from the file
    loader = TextLoader(file_path)
    documents=loader.load()
    
    #split the loades documents into chunks
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    docs=text_splitter.split_documents(documents)
    
    #creating the embeddings
    print("creating the embeddings")
    embeddings=OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("Finished creating the embeddings")
    
    #creating the vector store and persist it automatically
    #we have to pass the splitted or chunked data , embeddings and persistant directory
    db=Chroma.from_documents(
        docs,embeddings,persist_directory=persistant_directory
    )
    print("finished creating the vector store")
    
else:
    print("vector store already created.No need to initialize..")

    

