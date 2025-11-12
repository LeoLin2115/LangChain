from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser  
from dotenv import load_dotenv
import os

# class doing the output parsing 
class AnswerOutputParser(BaseOutputParser):
    def parse(self, text: str):
        ''' Parses the output of an LLM call'''
        return text.strip().split('answer =')
    


#set llm chat_prompt template, we use ollama here locally
llm = OllamaLLM(model="gemma2:2b")

# set the prompt template
# have two inputs system and human
# one variable : problem
template = '''You are a helpful assitant that solves math problems and shows your work.
            Output each step then return the answer in the following foarmat: answer = <answer here>.
            Make sure to output answer in all lowercases and to have exactly one space and one equal sigh following it.'''
human_template = '{problem}'

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", template),
     ("human", human_template),]
)

# format and run 
# give the variable problem
# run LLM
# parse output to answer
messages = chat_prompt.format_messages(problem = '2x^2 - 5x + 3 = 0')
result = llm.invoke(messages)
parsed = AnswerOutputParser().parse(result)
steps, answer = parsed
print(answer)
# print(steps)
# print(parsed)