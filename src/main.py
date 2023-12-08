import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from collections import Counter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from fastapi import FastAPI, Depends, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI()

os.environ["OPENAI_API_KEY"] = "user openai api key"

@app.get("/")
async def root():
    return {"message": "Hello World"}

def get_chatbot_response(chatbot_response):
    return(chatbot_response['result'].strip())


#로더 생성 후 문서 로드해오기
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader('data', glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
documents = loader.load()

#문서를 작은 단위로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=texts, embedding=embedding)
retriever = vectordb.as_retriever()

prompt_template ="""/
    You are an artificial intelligence Chatbot named "지웰봇" that specializes in summarizing 
    and answering documents about "경기도"'s youth welfare policy.
    {context}
    Question : {question}
    
    If the policy allows only residents of a specific city to sign up, it should not be recommended unless you specify the user's city.
    You must return in Korean. Return a accurate answer based on the document.
    Show the answer in a form that makes it look good for the user.
    If there is a url in the document, You have to answer as the url in the document.
    
    Answer : 
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["question"])

qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3),
    chain_type = "stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
    )

class Requestall(BaseModel):
    question : str

class Request(BaseModel):
    policy_name : str
    question : str

#기본 gpt
@app.post("/chat")
async def chat(request: Requestall):
    try:
        question = request.question

        chatbot_response = qa_chain(question)
        return JSONResponse(content={"answer": get_chatbot_response(chatbot_response)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#정책 domain이 정해진 상황에서 gpt에 질문할 때
@app.post("/policychat")
async def policychat(request: Request):
    try:
        policy_name = request.policy_name
        question = request.question

        string = f"{policy_name}에 관련된 정책입니다. "

        query = string + question
        chatbot_response = qa_chain(query)
        return JSONResponse(content={"answer": get_chatbot_response(chatbot_response)}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


