import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# --- All other imports are the same ---
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from supabase.client import Client, create_client

load_dotenv()
app = FastAPI()

origins = [
    "http://localhost:3000",
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

# Using the smaller model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str
    session_id: str

class UploadResponse(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str

# --- Helper Functions ---
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

# --- API Endpoints with DEBUGGING ADDED ---
@app.post("/upload", response_model=UploadResponse)
async def upload_files(pdf_docs: List[UploadFile] = File(...)):
    print("--- DEBUG: UPLOAD ENDPOINT HIT ---")
    if not pdf_docs:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")
    
    session_id = str(uuid.uuid4())
    
    try:
        print("--- DEBUG: STEP 1 - EXTRACTING TEXT FROM PDFS ---")
        raw_text = get_pdf_text(pdf_docs)
        print(f"--- DEBUG: STEP 1 SUCCESS - Extracted {len(raw_text)} characters ---")
        
        print("--- DEBUG: STEP 2 - CHUNKING TEXT ---")
        text_chunks = get_text_chunks(raw_text)
        print(f"--- DEBUG: STEP 2 SUCCESS - Created {len(text_chunks)} chunks ---")
        
        metadata = [{"session_id": session_id} for _ in text_chunks]
        
        print("--- DEBUG: STEP 3 - STORING EMBEDDINGS IN SUPABASE ---")
        SupabaseVectorStore.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            metadata=metadata
        )
        print("--- DEBUG: STEP 3 SUCCESS - Embeddings stored in Supabase ---")

        return {"message": "Files processed and stored successfully.", "session_id": session_id}
    except Exception as e:
        # If a normal Python error occurs, this will catch it and print it to the logs
        print(f"--- DEBUG: AN EXCEPTION OCCURRED ---")
        print(f"--- ERROR DETAILS: {str(e)} ---")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ... (The rest of your code, /chat and / endpoint, remains the same) ...
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents",
        )
        retriever = vector_store.as_retriever(
            search_kwargs={'filter': {'session_id': request.session_id}}
        )
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192", temperature=0.5
        )
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