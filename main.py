from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableSequence
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load PDF
loader = PyPDFLoader("C:/Users/USER/Desktop/python/langchain/LeoLin_SW_CV.pdf")
pages = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(pages)

# Create embeddings and vector store
# embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Define prompt template
prompt = PromptTemplate.from_template(
    "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
)

# Define LLM
llm = ChatOpenAI()

# Build chain using RunnableSequence
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question
query = "What is the main topic of this document?"
result = chain.invoke(query)
print(result)