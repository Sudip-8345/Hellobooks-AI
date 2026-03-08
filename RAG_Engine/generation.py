from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import config

def get_groq_llm():
    return ChatGroq(
        model=config.GROQ_MODEL, 
        temperature=config.LLM_TEMPERATURE, 
        api_key=config.GROQ_API_KEY
    )
    
def get_google_llm():
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GOOGLE_API_KEY
    )

def generate_response(query, context):
    prompt = f"""
    You are a helpful assistant. Answer the question based on the provided context.

    Question: {query}

    Context: {context}
    If the context does not contain the answer, say "I don't know based on the provided information."
    """
    try:
        llm = get_groq_llm()
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"[LLM] Groq model failed ({e}), trying fallback by Gemini...")
        try:
            llm = get_google_llm()
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"[LLM] Fallback model also failed ({e})")
            return "LLM is currently unavailable. Please try again later."
    