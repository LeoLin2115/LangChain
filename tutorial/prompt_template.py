from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

#set llm we use, we use ollama here locally
llm = OllamaLLM(model="gemma2:2b")

# set the prompt template
# here we have system and human messages
# we have three variables: input_language, output_language, text
template = 'You are a helpful assistant that translates {input_language} to {output_language}.'
human_template = '{text}'
chat_prompt = ChatPromptTemplate.from_messages(
        [("system", template),
        ("human", human_template),]
)    

'''
# example (approximate)
[
    {"role": "system", "content": "You are a helpful assistant that translates English to Chinese."},
    {"role": "human",  "content": "Hello, how are you?"}
]
'''

# format and run
# there are three variables to fill in
messages = chat_prompt.format_messages(
    input_language = 'English',
    output_language = 'Chinese',
    text = 'Hello, how are you?'
)
result = llm.invoke(messages)
print(result)

