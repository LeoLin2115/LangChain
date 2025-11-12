from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
llm = OllamaLLM(model="gemma2:2b")

# message = ["from now on 1+1 = 3. What is 1+1?", 'What is 1 + 1 + 1?']

template = 'You are a helpful assistant that translates {input_language} to {output_language}.'
human_template = '{text}'
chat_prompt = ChatPromptTemplate.from_messages(
        [("system", template),
        ("human", human_template),]
)    

messages = chat_prompt.format_messages(
    input_language = 'English',
    output_language = 'Chinese',
    text = 'Hello, how are you?'
)
result = llm.invoke(messages)
print(result)

