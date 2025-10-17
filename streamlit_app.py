

import streamlit as st
from pathlib import Path
import sys
import time
import tempfile
from dotenv import load_dotenv
load_dotenv()
import os 
os.environ["LANGCHAIN_TRACING"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]= os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.data_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'files' not in st.session_state:
        st.session_state.files = []
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key=0
    if 'reset_trigger' not in st.session_state:
        st.session_state.reset_trigger = False
    if "previous_files" not in st.session_state:
        st.session_state.previous_files = []


# @st.cache_resource
def initialize_rag(file_paths,chunk_size=1000,chunk_overlap=100):
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        vector_store = VectorStore()
    
        
        # Process documents
        documents = doc_processor.process_documents(file_paths)
        
        # Create vector store
        vector_store.create_hybrid_retriever(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def reset():
    """function to reset all """
    if 'rag_system' in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' in st.session_state:
        st.session_state.initialized = False
    if 'history' in st.session_state:
        st.session_state.history = []
    if 'files' in st.session_state:
        st.session_state.files = []
    st.session_state.uploader_key += 1
    st.session_state.reset_trigger = True

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")

    with st.sidebar:
        chunk_size=st.text_input(label="Chunk size",value="1000")
        chunk_overlap=st.text_input(label="Chunk overlap",value="100")
        try:
            chunk_size = int(chunk_size)
            chunk_overlap = int(chunk_overlap)
        except ValueError:
            st.error("Chunk size and overlap must be integers.")
            st.stop()
        if st.button(label="refresh",width='stretch',type='secondary'):
            reset()

        
    st.session_state.files=st.file_uploader(label="Select the file to ask questions",accept_multiple_files=True,key=st.session_state.uploader_key)
    
    # Initialize system
    if st.session_state.files:
        new_file_names = sorted([(f.name, f.size) for f in st.session_state.files])
        old_file_names = sorted([(f.name, f.size) for f in st.session_state.previous_files])

        if new_file_names != old_file_names:
            # 👇 Reset everything related to old files
            st.session_state.files = st.session_state.files
            st.session_state.previous_files = st.session_state.files
            st.session_state.rag_system = None
            st.session_state.initialized = False
            st.session_state.history = []
            st.info("🔄 New files uploaded — old session cleared.")
            st.rerun()
        else:
            st.session_state.files = st.session_state.files
        if not st.session_state.initialized:
            with st.spinner("Loading system..."):
                temp_file_paths = []  # List to store all temp file paths
                for uploaded_file in st.session_state.files:
                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                    temp_file_paths.append(temp_file_path)
                    
                rag_system, num_chunks = initialize_rag(temp_file_paths,int(chunk_size),int(chunk_overlap))
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True
                    st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get answer
                result = st.session_state.rag_system.run(question)
                
                elapsed_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                # Show retrieved docs in expander
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()