

##!pip install groq

##!pip install langchain_groq

##!pip install langchain

# Connecting The LLM Model To This Notebook Via It's Api
import streamlit as st
from langchain_groq import ChatGroq
import os

llm = ChatGroq(
    temperature=1,
    groq_api_key ='put_your_api_key_here',
    model_name = "llama-3.3-70b-versatile"
)
#response = llm.invoke("Hey Buddha")
#print(response.content) ### Here We Can print Response.metadata to see the metadata and also there are many other options just check them by printing response

# Setting up tools for extracting texts from the given URLS ( we will do this for text file, pdf )

## Using Webbase Loader

### Installing langchain_community package
##!pip install langchain_community

# Downloading And Installing The Embedding Model ( free and open source and the written model is good for llama llm model )

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings


#query = model.encode(" Hey ")

#print(len(query))    ## len(query) signifies that 384 features are used to do embedding of the text ## different models uses different numbers of features

##!pip install Playwright
##!pip install unstructured

working_dir = os.path.dirname(os.path.abspath(__file__))
vectordb_path = f"{working_dir}/rag_syatem/vectordatabase"


from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.document_loaders import PlaywrightURLLoader  ### For Capturing JS Websites
from langchain_community.document_loaders import PyPDFLoader
#url_path = ['https://www.amazon.in/Rotatable-Foldable-Tabletop-Adjustments-Smartphones/dp/B0CP93G4ZF/ref=sr_1_4?adgrpid=1329311913755935&dib=eyJ2IjoiMSJ9.GRRo4kl5QpZhrJnOfYQUHbz8rY__riGMDnNIQOv8hTpRQ0r2jwxu8y9Xq4wc44c7UWTM0_zsX_jZDmY_YSzsRnfZk_I50IOkRKi0N6tmXM5HUIi9zGG0ASIh4EgECPHMdFueg2qAEsZlgEDxNk1AMMx7to30RDIGl3tLrjiBrNa5QFOLAv4DiDa2SD9bIF7WmRAF2fgppUkOVhoAK80NEQO9VDqSVv9sw1QWdZrA8es.ksYjSSTqxtDyJavWZ1WWR07RgXJ1jxn06dQe347Lt8U&dib_tag=se&hvadid=83082222017808&hvbmt=be&hvdev=c&hvlocphy=148875&hvnetw=o&hvqmt=e&hvtargid=kwd-83083000420437%3Aloc-90&hydadcr=25202_2783765&keywords=amazon%2Bamazon%2Bmobile&mcid=5cc4e5c9c6723e4a86f8b746ad7735ff&msclkid=328efdd1fbfa1b08bb6494e788e46fb5&qid=1765176108&sr=8-4&th=1']
def url(uploaded_url:str):
    url_loader = WebBaseLoader(uploaded_url)
    url_content = url_loader.load()
    return url_content

## Addiing The Pdf Option Also In The Rag Application That We Are Building To Serve Peoples

#pdf_path = "/content/drive/MyDrive/RESUME/RESUME.pdf"
def pdf(uploaded_pdf):
    pdf_loader = PyPDFLoader(uploaded_pdf)
    pdf_content = pdf_loader.load()
    return pdf_content
#url_loader_2 = PlaywrightURLLoader(urls= url_path)
#url_content_2 = url_loader_2.load()

def pdf_chunker(pdf_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunked_pdf_content = text_splitter.split_documents(pdf_content)
    return chunked_pdf_content







#print(url_content)
#word_count = len(url_content[0].page_content.split())   ### Checkinig the number of words in the page_content of the url's webpage
#print(word_count)
#print(url_content)

#print(pdf_content)

# Now We Have To Chunk This Page Content Such That It Do Not Exceed The Token Limit Of The LLM Model

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunker(url_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50 )  #### We Can Add Metadata Here
    chunked_page_content = text_splitter.split_documents(url_content)
    return chunked_page_content




#print(len(chunked_page_content))
#print(len(chunked_pdf_content))

##!pip install chromadb

from langchain_core import documents
# Now Setting Up The Vector Data Base ( we are using chromadb for this project and can use other also )

from langchain_community.vectorstores import chroma
from langchain_community.vectorstores import Chroma

@st.cache_resource ## to ensure that the model is not loaded again and again that uses lazy loading and cause some issues while running the streamlit web application
def get_embedding_model():
    return SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embedding_model = get_embedding_model()
from langchain_core.documents import Document

def vectordatabasemanager():
    return Chroma(
        #documents=chunked_pdf_content,
        embedding_function=embedding_model,
        persist_directory=vectordb_path
        ### This Will Create A Persistent Local Disk File Which Will Store The EMbeddings In The PC And Work Locally Within The PC
    )



#total_chunks = chunked_page_content + chunked_pdf_content
def dbstoragemanager(chunked_page_content):
    ##model = SentenceTransformerEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'})
    ##vecdb = Chroma.from_documents(
    ##documents = chunked_page_content,
    ##embedding = embedding_model,
    ##persist_directory = r'C:\Users\91620\Desktop\rag_system\60e52b36-b764-495e-93fc-6d8ebb6e4b6f'   ### This Will Create A Persistent Local Disk File Which Will Store The EMbeddings In The PC And Work Locally Within The PC
    ##)
    ##return vecdb
    vecdb = vectordatabasemanager()
    vecdb.add_documents(chunked_page_content)
    vecdb.persist()
    return vecdb
def pdfvecdbmanager(chunked_pdf_content):
    ##vecdb = Chroma.from_documents(
        ##documents=chunked_pdf_content,
        ##embedding=embedding_model,
        ##persist_directory=r'C:\Users\91620\Desktop\rag_system\60e52b36-b764-495e-93fc-6d8ebb6e4b6f'
        ### This Will Create A Persistent Local Disk File Which Will Store The EMbeddings In The PC And Work Locally Within The PC
    ##)
    ##return vecdb
    vecdb = vectordatabasemanager()
    vecdb.add_documents(chunked_pdf_content)
    vecdb.persist()
    return vecdb


# Setting Up The Retriver Which Will Proceed towards the similarity search of query to the vector database

## Creating An Overall Fuction Which Will Contain All These Mini Functions And This Will Be Called By The Streamlit UI

def processing_manager(uploaded_url:str):
    url_content_1 = url(uploaded_url)
    chunked_page_content_1 = chunker(url_content_1)
    vecdb_1 = dbstoragemanager(chunked_page_content_1)
    return vecdb_1
def pdf_processing_manager(uploaded_pdf):
    pdf_content_1 = pdf(uploaded_pdf)
    chunked_pdf_content_1 = pdf_chunker(pdf_content_1)
    vecdb_1 = pdfvecdbmanager(chunked_pdf_content_1)
    return vecdb_1





## Testing The Cosine Similarity Search
#docs = retrieve_tool.invoke("car")
#print(len(docs)) ### Default number of chunk retrieved is 4

# Now Building A Pipeline Which Will Automatically Capture User Query And Send It To The Retriever Where It Is Embedded And Then Cosine Similarity Search is Used To Retrieve Most Closest Chunks And Then The Chunks Along With The Query Which Is Given By The User Is Given To The LLM Model With A Guiding Prompt Which Tells The LLM To Give Answers Based On The Given Content

from langchain.chains import RetrievalQA
##retrieve_tool = vecdb_1.as_retriever(search_kwargs={"k": 5})
##query_of_user = "what is the price of macbook ?"



# Building A Function System Where The Response Is Given In A Cleaner And Easily Readable Form

def structuring_response_manager(response):
    q = response['query']
    r = response['result']
    for source in response["source_documents"]:
        l = source.metadata['source']
    variable = r + "\n\n" + "Source Used For Generating The Answer" + "\n" +l
    return variable



# Testing The Pipeline By Giving It The Query Of User And Storing The Response In A Variable And Printing It

##llm_response = retrieval_chain(query_of_user)
##structured_response(llm_response)       ### The LLM Response Is A Dictionary Having Keys As =>>>>> "query", "result", "source_documents=Document[metadata = source, title and description,  page_content], Document[simiar structure], ....""

## Now Building A Pipeline That Will Be Used When User Give The Query And Press Get The Answer Button In Our Streamlit WebApplication






def get_the_answer(created_vecdb_1, user_query):
    retriever_manager = created_vecdb_1.as_retriever(search_kwargs={"k": 5})

    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever = retriever_manager,
        return_source_documents=True
    )

    response = retrieval_chain(user_query)
    good_response = structuring_response_manager(response)
    return good_response

#print(llm_response)

# The Retrieval Augmented Generation System Is Working Fine

# Now We Will Create A UI And Connect This Backend To It

# Then We Will Dockerize The Application

# And We Will Deploy It On The Cloud Platform Such That It Is Accessible Through Online Search Also