from RAG_Engine.indexing import load_documents, chunk_docs, load_vectorstore
from RAG_Engine.retrieval import get_retriever, rerank_chunks, get_source_name
from RAG_Engine.generation import generate_response

docs = load_documents("hellobooks_dataset")
chunks = chunk_docs(docs)
vectorstore = load_vectorstore(chunks)

retriever = get_retriever(vectorstore=vectorstore)

def answer(query: str) -> str:
    """Run the RAG pipeline for a single query and return the answer."""
    raw_context = retriever.invoke(query)
    doc_tuples = [(doc, i) for i, doc in enumerate(raw_context)]
    reranked = rerank_chunks(query, doc_tuples)
    context = "\n\n".join(
        [f"Source: {get_source_name(doc)}\nContent: {doc.page_content}" for doc, _ in reranked]
    )
    response = generate_response(query, context)
    return response