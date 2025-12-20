import os
from dotenv import load_dotenv
from langchain_google_genai import (GoogleGenerativeAI, GoogleGenerativeAIEmbeddings)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ENV
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY

# VECTOR STORE
def get_vectorstore(persist_dir="chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004"
    )
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

def get_retriever(vs, k=4):
    return vs.as_retriever(search_kwargs={"k": k})

# LLM
def get_llm():
    return GoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

# PROMPT
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer ONLY using the given context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

# BUILD RAG CHAIN 
def build_rag_chain():
    vectorstore = get_vectorstore()
    retriever = get_retriever(vectorstore)
    llm = get_llm()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain

# FORMAT ANSWER 
def format_answer(answer):
    return answer



