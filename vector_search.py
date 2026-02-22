from supabase_client import get_supabase
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def semantic_search(query):
    supabase = get_supabase()
    qvec = embeddings_model.embed_query(query)
    vec_str = str(qvec)

    sql = f"""
    SELECT tc.chunk_text
    FROM embeddings e
    JOIN text_chunks tc ON e.chunk_id = tc.id
    ORDER BY e.embedding <-> '{vec_str}'::vector
    LIMIT 5;
    """

    return supabase.rpc("execute_sql", {"query": sql}).execute().data
