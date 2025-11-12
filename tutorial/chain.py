from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser  
from dotenv import load_dotenv
import os

# class doing the output parsing 
class CommaSeparatedListFiveOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(', ')
    


#set llm chat_prompt template, we use ollama here locally
chat_model = OllamaLLM(model="gemma2:2b")

# set the prompt template
# have two inputs system and human
# one variable : text
template = '''You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more.
'''
human_template = '{text}'

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", template),
     ("human", human_template),]
)

# chain three things: prompt, llm, output parser
# format and run, one variable text
chain = chat_prompt  | CommaSeparatedListFiveOutputParser() | chat_model
result = chain.invoke({'text': 'colors'})
print(result)
# print(steps)
# print(parsed)