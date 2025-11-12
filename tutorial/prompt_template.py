from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

#set llm we use, we use ollama here locally
llm = OllamaLLM(model="gemma2:2b")

# set the prompt template

chat_prompt = ChatPromptTemplate.from_messages(
        [("system", template),
        ("human", human_template),]
)    
template = 'You are a helpful assistant that translates {input_language} to {output_language}.'
human_template = '{text}'


messages = chat_prompt.format_messages(
    input_language = 'English',
    output_language = 'Chinese',
    text = 'Hello, how are you?'
)
result = llm.invoke(messages)
print(result)

