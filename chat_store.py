from supabase_client import get_supabase

def save_message(session_id, role, message):
    supabase = get_supabase()
    supabase.table("chat_messages").insert({
        "session_id": session_id,
        "role": role,
        "message": message
    }).execute()
