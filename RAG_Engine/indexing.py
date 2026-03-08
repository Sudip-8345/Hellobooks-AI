import os
import config
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_documents(dir = "./hellobooks_dataset"):
    loader = DirectoryLoader(
        path=dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"Total {len(docs)} are loaded.")
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n---", "\n\n", "\n", ". ", " ", ""],
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"The docs is splitted into {len(chunks)} chunks")
    return chunks

embedding = GoogleGenerativeAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    api_key=config.GOOGLE_API_KEY
)

def load_vectorstore(chunks, vectordb = config.FAISS_PERSIST_DIR):
    if not os.path.exists(vectordb):
        store = FAISS.from_documents(chunks, embedding)
        store.save_local(vectordb)
        print(f"FAISS vectorstore is created and saved at {vectordb}")
        
    else:
        store = FAISS.load_local(vectordb, embedding, allow_dangerous_deserialization=True)
        print(f"FAISS vectorstore is loaded from {vectordb}")
    return store
