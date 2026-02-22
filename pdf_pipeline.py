from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from supabase_client import get_supabase

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(file, user_id):
    supabase = get_supabase()

    # Upload PDF to storage
    path = f"{user_id}/{file.name}"
    supabase.storage.from_("pdf-files").upload(path, file.read())

    url = supabase.storage.from_("pdf-files").get_public_url(path)

    # Save PDF record
    pdf = supabase.table("pdf_documents").insert({
        "user_id": user_id,
        "file_name": file.name,
        "file_url": url
    }).execute().data[0]

    # Read text
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        if p.extract_text():
            text += p.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_text(text)

    # Save chunks + embeddings
    for i, chunk in enumerate(chunks):
        page = supabase.table("pdf_pages").insert({
            "pdf_id": pdf["id"],
            "page_number": i,
            "content": chunk
        }).execute().data[0]

        chunk_row = supabase.table("text_chunks").insert({
            "page_id": page["id"],
            "chunk_text": chunk
        }).execute().data[0]

        vec = embeddings_model.embed_query(chunk)

        supabase.table("embeddings").insert({
            "chunk_id": chunk_row["id"],
            "embedding": vec
        }).execute()

    return pdf["id"]
