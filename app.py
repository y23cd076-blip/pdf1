import streamlit as st
from pdf_pipeline import process_pdf
from vector_search import semantic_search
from chat_store import save_message
from image_store import upload_image
from supabase_client import get_supabase
import uuid

st.set_page_config("SlideSense AI", "📘", layout="wide")

supabase = get_supabase()

# fake user for now (later replace with Supabase Auth)
USER_ID = "demo-user"
SESSION_ID = str(uuid.uuid4())

st.title("📘 SlideSense AI Cloud Platform")

mode = st.sidebar.radio("Mode", ["PDF AI", "Image AI"])

# ================= PDF =================
if mode == "PDF AI":
    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf:
        with st.spinner("Processing PDF & building vector DB..."):
            pdf_id = process_pdf(pdf, USER_ID)
            st.success("PDF processed and stored in cloud DB")

    q = st.text_input("Ask question")
    if q:
        results = semantic_search(q)
        context = "\n".join([r["chunk_text"] for r in results])

        answer = f"Answer from context:\n{context[:1000]}"

        save_message(SESSION_ID, "user", q)
        save_message(SESSION_ID, "ai", answer)

        st.markdown("### 🤖 AI Answer")
        st.write(answer)

# ================= IMAGE =================
if mode == "Image AI":
    img = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    if img:
        img_id, url = upload_image(img, USER_ID)
        st.image(url)
        st.success("Image stored in cloud")

        q = st.text_input("Ask question about image")
        if q:
            st.write("AI Answer: (connect BLIP/Gemini here)")
