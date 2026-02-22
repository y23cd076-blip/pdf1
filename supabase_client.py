from supabase import create_client
import os

def get_supabase():
    return create_client(
        os.getenv("SUPABASE_URL"),       # env variable name
        os.getenv("SUPABASE_ANON_KEY")   # env variable name
    )
