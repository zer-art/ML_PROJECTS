# to push or update the knowledge base in pinecone vector store
from src.functions import load_pdf_file , text_split , download_hugging_face
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os 

load_dotenv()
pinecode_api = os.environ.get('Pinecode_API')
gemini_api = os.environ.get('Gemini_API')

#setting api as local variable
os.environ["GEMINI_API_KEY"] = gemini_api
os.environ["PINECONE_API_KEY"] = pinecode_api 


extracted_data = load_pdf_file("Data")
text_chunks = text_split(extracted_data)


docsearch = PineconeVectorStore.from_documents(
    text_chunks,
    embedding=download_hugging_face(),
    index_name="healthcare-bot"
    
) 