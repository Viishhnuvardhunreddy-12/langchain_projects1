from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
llm=ChatOpenAI(model="gpt-3.5-turbo")
#summary template
summary_template=ChatPromptTemplate.from_messages([
    ("system","you are a movie critic."),
    ("user","provide a brief explanation on {movie_name}"),
])

#Define plot analysis step
def analyse_plot(plot):
    plot_template=ChatPromptTemplate.from_messages([
        ("system","you are a movie critic."),
        ("user","Analyse the plot {plot}. what are the strengths and weeknesses?"),
    ])
    return plot_template.format_prompt(plot=plot)

#Define the character analysis step
def analyse_character(character):
    character_template=ChatPromptTemplate.from_messages([
        ("system","you a movie critic."),
        ("user","Analyse the characters {characters}.what are their strengths and weeknesses."),
    ])
    return character_template.format_prompt(character=character)

#combine verdicts
def combine_verdicts(plot_analysis,character_analysis):
    return f"plot analysis:\n{plot_analysis}\n\ncharacter analysis:\n{character_analysis}"

# simplify the batches
plot_branch_chain=(
    RunnableLambda(lambda x:analyse_plot(x)) | llm | StrOutputParser()
)

character_branch_chain=(
    RunnableLambda(lambda x:analyse_character(x)) | llm | StrOutputParser()
)

chain=(
    summary_template
    | llm
    | StrOutputParser
    | RunnableParallel(branches={"plot":plot_branch_chain,"characters":character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"],x["branches"]["character"]))
)   
result=chain.invoke({"movie_name":"Annabelle"})
print(result)
