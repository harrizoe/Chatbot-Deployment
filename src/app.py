def query_rag(user_query: str) -> str:
    # Load models (embedding, reranker, LLM) — consider caching for performance
    embedding_model = load_embedding_model()
    reranker_model = load_reranker_model()
    phi_pipeline = load_phi_pipeline()

    # Connect to Pinecone
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = "unhcr-chatbot"
    index = initialize_pinecone(pinecone_api_key, index_name, dimension=768)

    # RAG steps
    reranked_docs = store_retrieve_and_rerank(user_query, index, embedding_model, reranker_model)
    
    # Prompt template
    prompt_template = """SYSTEM: You are a chatbot specializing in UNHCR and refugee services in Denmark. 
Answer the user’s question concisely and directly using the provided CONTEXT. 
If the question is not related to UNHCR or refugee services, respond with: "I'm sorry, I can only answer questions related to UNHCR and refugee services in Denmark."
If the answer is incomplete (e.g., if a question asks for detailed steps, like "What happens after I apply for asylum?"), provide as many relevant details as possible.
At the end of your answer, if an answer is available, add a note: "For additional information, please visit: {sources}"

CONTEXT: {context}

USER QUERY: {query}

INSTRUCTIONS:
- Answer concisely in 2-3 sentences.
- Include detailed steps if applicable.
- Do NOT generate follow-up questions.
- Do NOT continue generating after your answer.
- If unsure or off-topic, reply with: "I'm sorry, I can only answer questions related to UNHCR and refugee services in Denmark."

FINAL RESPONSE:
"""
    answer = generate_answer(user_query, reranked_docs, prompt_template, phi_pipeline)
    return answer

