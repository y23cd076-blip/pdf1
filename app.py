import streamlit as st
import requests
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import time
import os

# -------------------- Page Config --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

# -------------------- Custom CSS Animations --------------------
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .animated-title {
        animation: fadeIn 1s ease-out;
        font-size: 3em;
        text-align: center;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .login-box {
        animation: slideIn 0.8s ease-out;
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .pulse-icon {
        animation: pulse 2s infinite;
        font-size: 5em;
        text-align: center;
    }
    
    .rotate-icon {
        animation: rotate 3s linear infinite;
        font-size: 3em;
        display: inline-block;
    }
    
    .feature-card {
        animation: fadeIn 1.2s ease-out;
        padding: 1.5rem;
        border-radius: 10px;
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Animation Functions --------------------
def show_welcome_animation():
    """Display welcome animation with emojis"""
    welcome = st.empty()
    icons = ["🚀", "✨", "🎯", "💡", "📚", "🔥"]
    for icon in icons:
        welcome.markdown(f'<div class="pulse-icon">{icon}</div>', unsafe_allow_html=True)
        time.sleep(0.3)
    welcome.empty()

def show_success_animation():
    """Show success animation"""
    st.balloons()
    time.sleep(0.5)

# -------------------- Check API Key --------------------
if "GOOGLE_API_KEY" not in os.environ:
    st.error("🔑 Missing GOOGLE_API_KEY! Please add it to Streamlit secrets or environment variables.")
    st.info("Go to App Settings → Secrets and add: GOOGLE_API_KEY = 'your_key_here'")
    st.stop()

# -------------------- Session Defaults --------------------
defaults = {
    "chat_history": [],
    "vector_db": None,
    "authenticated": False,
    "users": {"admin": "admin123"}
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- Authentication UI --------------------
def login_ui():
    # Animated Welcome Title
    st.markdown('<h1 class="animated-title">🔐 SlideSense</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        # Animated login icon
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <div class="pulse-icon">🔐</div>
            <h2 style="color: #667eea;">Secure Login</h2>
            <p style="color: #666;">AI-Powered Learning Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards with animation
        st.markdown("""
        <div class="feature-card">
            📘 <strong>PDF Analysis</strong><br>
            <small>Ask questions about your documents</small>
        </div>
        <div class="feature-card">
            🖼 <strong>Image Q&A</strong><br>
            <small>Visual question answering</small>
        </div>
        <div class="feature-card">
            🤖 <strong>AI Powered</strong><br>
            <small>Gemini & BLIP technology</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown("### 🚀 Get Started")

        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])

        with tab1:
            u = st.text_input("Username", key="login_user", placeholder="Enter username")
            p = st.text_input("Password", type="password", key="login_pass", placeholder="Enter password")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
            with col_btn2:
                if st.button("🚀 Login", use_container_width=True):
                    if u in st.session_state.users and st.session_state.users[u] == p:
                        with st.spinner("Logging in..."):
                            time.sleep(0.5)
                        st.success("Login Successful! 🎉")
                        show_success_animation()
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Invalid credentials ❌")
            
            st.info("💡 Default: admin / admin123")

        with tab2:
            nu = st.text_input("New Username", key="signup_user", placeholder="Choose username")
            np = st.text_input("New Password", type="password", key="signup_pass", placeholder="Choose password")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
            with col_btn2:
                if st.button("✨ Create Account", use_container_width=True):
                    if not nu or not np:
                        st.warning("Please fill all fields")
                    elif nu in st.session_state.users:
                        st.warning("User already exists")
                    else:
                        with st.spinner("Creating account..."):
                            time.sleep(0.5)
                        st.session_state.users[nu] = np
                        st.success("Account created! 🎉")
                        show_success_animation()
        
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Load BLIP VQA --------------------
@st.cache_resource
def load_blip_vqa():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        return processor, model
    except Exception as e:
        st.error(f"❌ Failed to load BLIP model: {str(e)}")
        return None, None

processor, blip_vqa_model = load_blip_vqa()

# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question):
    try:
        if processor is None or blip_vqa_model is None:
            return "⚠️ Image analysis model not available. Please check model loading."
        
        # Step 1: BLIP short answer
        inputs = processor(image, question, return_tensors="pt")

        output = blip_vqa_model.generate(
            **inputs,
            max_length=10,
            num_beams=5,
            early_stopping=True
        )

        short_answer = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        # Step 2: Expand using Gemini (text-only)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

        expansion_prompt = f"""
You are expanding a visual answer.

Image Question:
{question}

Vision Model Answer:
{short_answer}

Task:
- Convert this into a clear, complete sentence.
- Do NOT add extra details.
- Keep accurate.
"""

        final_answer = llm.invoke(expansion_prompt)
        return final_answer.content
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"

# -------------------- Auth Check --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.markdown('<p style="animation: fadeIn 1s ease-out;">✅ <strong>Logged in successfully!</strong></p>', unsafe_allow_html=True)

# Cache clearing tool
st.sidebar.markdown("### 🗑️ Cache Management")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔄 Clear Cache"):
        with st.spinner("Clearing cache..."):
            st.cache_resource.clear()
            st.session_state.vector_db = None
            time.sleep(0.5)
        st.sidebar.success("✅ Cleared!")
        st.snow()
        time.sleep(1)
        st.rerun()

with col2:
    if st.button("🗑️ Clear History"):
        with st.spinner("Clearing history..."):
            st.session_state.chat_history = []
            time.sleep(0.5)
        st.sidebar.success("✅ Cleared!")
        time.sleep(1)
        st.rerun()

if st.sidebar.button("🚪 Logout", use_container_width=True):
    with st.spinner("Logging out..."):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.cache_resource.clear()
        time.sleep(0.5)
    st.sidebar.success("👋 Goodbye!")
    time.sleep(0.5)
    st.rerun()

st.sidebar.divider()
page = st.sidebar.radio("🎯 Select Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

st.sidebar.divider()
st.sidebar.markdown("### 💬 Recent History")
if st.session_state.chat_history:
    for idx, (q, a) in enumerate(st.session_state.chat_history[-6:]):
        st.sidebar.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.3rem 0; background: rgba(102, 126, 234, 0.1); border-radius: 5px; font-size: 0.85em;">
            {idx + 1}. {q[:30]}...
        </div>
        """, unsafe_allow_html=True)
else:
    st.sidebar.info("No history yet. Start chatting!")

# -------------------- Hero --------------------
st.markdown('<h1 class="animated-title">📘 SlideSense AI Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">Smart Learning | Smart Vision | Smart AI</p>', unsafe_allow_html=True)

# Animated progress bar on first load
if 'first_load' not in st.session_state:
    st.session_state.first_load = True
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    progress_bar.empty()
    st.balloons()

st.divider()

# -------------------- PDF ANALYZER --------------------
if page == "📘 PDF Analyzer":
    st.markdown("## 📄 Upload Your PDF Document")
    
    # Animated upload section
    st.markdown("""
    <div class="feature-card">
        <div style="text-align: center;">
            <span class="rotate-icon">📤</span>
            <p>Drag and drop your PDF file or click to browse</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    pdf = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

    if pdf:
        if st.session_state.vector_db is None:
            # Animated processing
            with st.spinner("🧠 Processing PDF..."):
                progress_bar = st.progress(0)
                try:
                    reader = PdfReader(pdf)
                    progress_bar.progress(20)
                    
                    text = ""
                    for page in reader.pages:
                        if page.extract_text():
                            text += page.extract_text() + "\n"
                    progress_bar.progress(40)

                    if not text.strip():
                        st.error("❌ No text found in PDF. The PDF might be image-based or empty.")
                        st.stop()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=80
                    )
                    chunks = splitter.split_text(text)
                    progress_bar.progress(60)

                    try:
                        asyncio.get_running_loop()
                    except:
                        asyncio.set_event_loop(asyncio.new_event_loop())

                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    progress_bar.progress(80)

                    st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)
                    progress_bar.progress(100)
                    time.sleep(0.3)
                    progress_bar.empty()
                    
                    st.success("✅ PDF processed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.stop()

        # Success indicator with animation
        st.markdown("""
        <div class="feature-card" style="background: rgba(34, 197, 94, 0.1); border-left-color: #22c55e;">
            ✅ <strong>PDF Ready!</strong> Ask any question about your document below.
        </div>
        """, unsafe_allow_html=True)
        
        q = st.text_input("💭 Ask your question", placeholder="What is this document about?")

        if q:
            with st.spinner("🤖 AI Thinking..."):
                try:
                    docs = st.session_state.vector_db.similarity_search(q, k=5)
                    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

                    history = ""
                    for x, y in st.session_state.chat_history[-5:]:
                        history += f"Q:{x}\nA:{y}\n"

                    prompt = ChatPromptTemplate.from_template("""
History:
{history}

Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

                    chain = create_stuff_documents_chain(llm, prompt)
                    res = chain.invoke({
                        "context": docs,
                        "question": q,
                        "history": history
                    })

                    st.session_state.chat_history.append((q, res))
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

        if st.session_state.chat_history:
            st.markdown("## 💬 Conversation History")
            for idx, (q, a) in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"""
                    <div class="feature-card">
                        <strong>🧑 Question {len(st.session_state.chat_history) - idx}:</strong><br>
                        {q}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="feature-card" style="background: rgba(102, 126, 234, 0.05);">
                        <strong>🤖 Answer:</strong><br>
                        {a}
                    </div>
                    """, unsafe_allow_html=True)
                    st.divider()

# -------------------- IMAGE QUESTION ANSWERING --------------------
if page == "🖼 Image Q&A":
    st.markdown("## 🖼️ Visual Question Answering")
    
    # Animated upload section
    st.markdown("""
    <div class="feature-card">
        <div style="text-align: center;">
            <span class="rotate-icon">🎨</span>
            <p>Upload an image and ask questions about it!</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if img_file:
        try:
            img = Image.open(img_file).convert("RGB")
            
            # Display image with animation
            with st.container():
                st.markdown('<div style="animation: fadeIn 0.8s ease-out;">', unsafe_allow_html=True)
                st.image(img, use_column_width=True, caption="Uploaded Image")
                st.markdown('</div>', unsafe_allow_html=True)

            question = st.text_input("💭 Ask a question about the image", placeholder="What do you see in this image?")

            if question:
                # Animated thinking process
                with st.spinner("🤖 Analyzing image..."):
                    progress_bar = st.progress(0)
                    
                    for i in range(30):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    answer = answer_image_question(img, question)
                    
                    for i in range(30, 100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    progress_bar.empty()

                # Display answer with animation
                st.markdown(f"""
                <div class="feature-card" style="background: rgba(34, 197, 94, 0.1); border-left-color: #22c55e; animation: slideIn 0.5s ease-out;">
                    <strong>🤖 AI Response:</strong><br>
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
                
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
