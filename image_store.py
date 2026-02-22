from supabase_client import get_supabase

def upload_image(file, user_id):
    supabase = get_supabase()
    path = f"{user_id}/{file.name}"
    supabase.storage.from_("image-files").upload(path, file.read())
    url = supabase.storage.from_("image-files").get_public_url(path)

    row = supabase.table("image_uploads").insert({
        "user_id": user_id,
        "file_url": url
    }).execute().data[0]

    return row["id"], url
