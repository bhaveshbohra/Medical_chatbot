from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from src.extra_docs import get_extra_docs  ## for new doc data 

# Load environment
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# # Load PDFs + custom docs( with extra document)
# extracted_data = load_pdf_file(data="data/")
# filter_data = filter_to_minimal_docs(extracted_data) ## this line for extra add 

# custom_docs = get_extra_docs()
# all_docs = filter_data + custom_docs

# # Split into chunks
# text_chunks = text_split(all_docs)


# Load PDFs (without extra documents)
extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

# Get embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone setup
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# OPTION 1: Fresh index (overwrite)
def create_new_index(docs):

    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        index_name=index_name,
        embedding=embeddings,
    )

    return vectorstore

# -------------------------------
# OPTION 2: Append to existing index
def append_to_existing_index(docs):

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    vectorstore.add_documents(docs)

    return vectorstore

if __name__ == "__main__":

    
    # Fresh index (use carefully, wipes old data)
    # docsearch = create_new_index(text_chunks)
    
    # Append mode (recommended)
    docsearch = append_to_existing_index(text_chunks)



