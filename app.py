# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from pydantic import BaseModel
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os

# # Initialize FastAPI app
# app = FastAPI()

# # Mount static files (CSS/JS under /static)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Jinja2 templates (for rendering chat.html)
# templates = Jinja2Templates(directory="templates")

# # Load environment variables
# load_dotenv()
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# # Setup embeddings + Pinecone
# embeddings = download_hugging_face_embeddings()
# index_name = "medical-chatbot"

# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# chatModel = ChatOpenAI(model="gpt-4o")

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# # Input schema for JSON requests
# class ChatRequest(BaseModel):
#     msg: str


# # Route: Home page
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("chat.html", {"request": request})


# # Route: Chat API (JSON request/response)
# @app.post("/get")
# async def chat(req: ChatRequest):
#     response = rag_chain.invoke({"input": req.msg})
#     return JSONResponse({"answer": response["answer"]})


# # To run: uvicorn main:app --host 0.0.0.0 --port 8080 --reload


from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)