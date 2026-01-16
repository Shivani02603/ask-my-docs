import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil
import subprocess
import sys
import tempfile

# Page config
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide"
)

CHROMA_PATH = "chroma"
DATA_PATH = "data"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def get_embedding_function():
    """Get embedding function - using free HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

def initialize_database():
    """Initialize the database by running setup_database.py"""
    if not os.path.exists(CHROMA_PATH):
        with st.spinner("🔄 Initializing database for the first time... This may take a minute."):
            try:
                # Run setup_database.py
                result = subprocess.run(
                    [sys.executable, "setup_database.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.success("✅ Database initialized successfully!")
                return True
            except subprocess.CalledProcessError as e:
                st.error(f"❌ Failed to initialize database: {e.stderr}")
                return False
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return False
    return True

def process_uploaded_pdfs(uploaded_files):
    """Process uploaded PDF files and create/update the database"""
    try:
        # Clear existing database
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        all_chunks = []
        
        with st.spinner(f"📄 Processing {len(uploaded_files)} PDF(s)..."):
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load and split the PDF
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=80,
                    length_function=len,
                )
                chunks = text_splitter.split_documents(documents)
                
                # Add source filename to metadata
                for chunk in chunks:
                    chunk.metadata['source'] = uploaded_file.name
                
                all_chunks.extend(chunks)
                
                # Clean up temp file
                os.unlink(tmp_path)
        
        # Create new database with all chunks
        with st.spinner(f"💾 Creating database with {len(all_chunks)} chunks..."):
            db = Chroma.from_documents(
                all_chunks,
                get_embedding_function(),
                persist_directory=CHROMA_PATH
            )
            db.persist()
        
        return True, len(all_chunks)
    
    except Exception as e:
        return False, str(e)

def query_rag(query_text: str, api_key: str = None):
    """Query the RAG system"""
    try:
        # Prepare the DB
        embedding_function = get_embedding_function()
        
        if not os.path.exists(CHROMA_PATH):
            # Try to initialize automatically
            if not initialize_database():
                return "⚠️ Database initialization failed. Please check the logs.", []
            return "No relevant documents found.", []
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Use OpenAI if API key provided, otherwise return context only
        if api_key:
            try:
                model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0)
                response_text = model.invoke(prompt).content
            except Exception as e:
                response_text = f"⚠️ Error with OpenAI API: {str(e)}\n\nRelevant context found:\n{context_text}"
        else:
            response_text = f"💡 No API key provided. Here's the relevant context:\n\n{context_text}"
        
        sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
        
        return response_text, sources
        
    except Exception as e:
        return f"❌ Error: {str(e)}", []

def main():
    st.title("📚 RAG Document Q&A System")
    st.markdown("Ask questions about your documents using AI-powered search")

    # Sidebar
    with st.sidebar:
        st.header("📤 Upload PDFs")
        st.markdown("Upload your own PDF files to ask questions about them!")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Uploaded PDFs", type="primary"):
                success, result = process_uploaded_pdfs(uploaded_files)
                if success:
                    st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s) with {result} chunks!")
                    st.balloons()
                else:
                    st.error(f"❌ Error: {result}")
        
        st.divider()
        st.header("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key (Optional)", 
            type="password",
            help="Provide your OpenAI API key for GPT-powered answers. Leave blank to see context only."
        )
        
        st.divider()
        
        st.subheader("📊 Database Status")
        if os.path.exists(CHROMA_PATH):
            try:
                db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
                doc_count = db._collection.count()
                st.success(f"✅ Database active\n\n📄 {doc_count} chunks indexed")
            except:
                st.warning("⚠️ Database exists but may need reinitialization")
        else:
            st.error("❌ Database not initialized")
        
        st.divider()
        
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. **With OpenAI API**: Enter your API key above for AI-generated answers
        2. **Without API**: Get relevant document excerpts only
        3. Type your question in the main area
        4. Click 'Search' to get answers
        """)
        
        st.divider()
        st.caption("Built with LangChain + ChromaDB + Streamlit")
    
    # Main area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Your Question:",
            placeholder="e.g., What are the rules of Monopoly?",
            height=100
        )
    
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner("🔎 Searching documents..."):
            response, sources = query_rag(query, api_key)
        
        st.divider()
        
        # Display response
        st.subheader("📝 Answer")
        st.markdown(response)
        
        # Display sources
        if sources:
            st.divider()
            st.subheader("📎 Sources")
            for i, source in enumerate(sources, 1):
                st.caption(f"{i}. {source}")
    
    elif search_button:
        st.warning("⚠️ Please enter a question first")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        Try asking:
        - What are the basic rules?
        - How do players win?
        - What happens when you land on a property?
        - Explain the game setup
        """)

if __name__ == "__main__":
    main()
