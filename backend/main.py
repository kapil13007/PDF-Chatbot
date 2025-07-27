import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from supabase.client import Client, create_client

# Load environment variables from .env file
load_dotenv()

# --- Initialize FastAPI App and CORS ---
app = FastAPI()

# IMPORTANT: When you deploy your frontend, add its URL here
origins = [
    "http://localhost:3000", # For local React development
    # e.g., "https://my-pdf-chat-app.vercel.app" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Supabase and Embeddings ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    raise Exception("Supabase URL and Key must be set in the environment variables.")

supabase: Client = create_client(supabase_url, supabase_key)
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

# --- Pydantic Models for API requests/responses ---
class ChatRequest(BaseModel):
    question: str
    session_id: str

class UploadResponse(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str

# --- Helper Functions (No changes here) ---
def get_pdf_text(pdf_docs: List[UploadFile]) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(raw_text: str) -> List[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(raw_text)

# --- Updated API Endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload_files(pdf_docs: List[UploadFile] = File(...)):
    if not pdf_docs:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")
    
    # This ID will link all chunks from this upload session together
    session_id = str(uuid.uuid4())
    
    try:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        
        # Add the session_id to each chunk's metadata
        metadata = [{"session_id": session_id} for _ in text_chunks]
        
        # Store embeddings in the 'documents' table in Supabase
        SupabaseVectorStore.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents", # This is the DB function we created
            metadata=metadata
        )

        return {"message": "Files processed and stored successfully.", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Initialize the vector store to retrieve data from Supabase
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents",
        )

        # Create a retriever that ONLY looks for documents matching the specific session_id
        retriever = vector_store.as_retriever(
            search_kwargs={'filter': {'session_id': request.session_id}}
        )

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192", temperature=0.5
        )
        
        # Create the conversation chain on-the-fly for each request
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        )
        
        response = conversation_chain.invoke({'question': request.question})
        return {"answer": response['answer']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during conversation: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "PDF Chatbot API is running."}