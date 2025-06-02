from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from src.prompt import system_prompt 
import os 
from dotenv import load_dotenv

load_dotenv()
pinecode_api = os.environ.get('Pinecode_API')
gemini_api = os.environ.get('Gemini_API')

#setting api as local variable
os.environ["GEMINI_API_KEY"] = gemini_api
os.environ["PINECONE_API_KEY"] = pinecode_api 


# pdf loader
def load_pdf_file (data): 
    loader = DirectoryLoader( 
        data,
        glob= "*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# convert data into text chunks
def text_split(extracted_data): 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 20 )
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk

# get embeddings
def download_hugging_face(): 
    embd  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embd


def load_rag_chain():
    embd = download_hugging_face()
    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embd,
        index_name="healthcare-bot"
    )
    retriver  = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        max_output_tokens=1024,
        top_p=0.95,
        top_k=40,
        google_api_key=gemini_api
    )

    qa_chain = create_stuff_documents_chain(llm=gemini, prompt=prompt)
    rag = create_retrieval_chain(retriver, qa_chain)
    return rag

