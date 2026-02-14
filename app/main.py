import streamlit as st
import os
import sys

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.retrieval import get_vector_store, get_qa_chain
    from src.query_engine import CricketQueryEngine
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Cricket Stats Bot", 
    layout="wide", 
    page_icon="C",
    initial_sidebar_state="expanded"
)

# Light Blue Theme CSS
st.markdown("""
<style>
    /* Comfortable light blue background */
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    
    /* Main content */
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Header */
    .cricket-header {
        text-align: center;
        padding: 2rem 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(25, 118, 210, 0.2);
    }
    
    .cricket-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .cricket-subtitle {
        color: #e3f2fd;
        font-size: 1rem;
    }
    
    /* Sponsor banner */
    .sponsor-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 2px solid #e3f2fd;
        text-align: center;
    }
    
    .sponsor-title {
        color: #1976d2;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .sponsor-logos {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .sponsor-item {
        background: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: 1px solid #e3f2fd;
        font-weight: 600;
        color: #1976d2;
        font-size: 0.9rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #42a5f5;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e3f2fd 0%, #ffffff 100%);
        border-right: 1px solid #bbdefb;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.9rem;
        width: 100%;
        text-align: left;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(25, 118, 210, 0.2);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
        box-shadow: 0 4px 8px rgba(25, 118, 210, 0.3);
        transform: translateY(-1px);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #e3f2fd;
        border-radius: 8px;
        font-weight: 600;
        color: #1976d2;
        padding: 0.5rem;
    }
    
    /* Input */
    .stChatInputContainer {
        border-top: 2px solid #e3f2fd;
        padding-top: 1rem;
        background-color: white;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1976d2;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #546e7a;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #e3f2fd 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #bbdefb;
        box-shadow: 0 2px 6px rgba(25, 118, 210, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1976d2;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #546e7a;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="cricket-header">
    <div class="cricket-title">Cricket Stats Bot</div>
    <div class="cricket-subtitle">Your AI-powered cricket statistics expert - ODI | T20 | Test</div>
</div>
""", unsafe_allow_html=True)

# Sponsor Banner
st.markdown("""
<div class="sponsor-banner">
    <div class="sponsor-title">POWERED BY OUR SPONSORS</div>
    <div class="sponsor-logos">
        <div class="sponsor-item">CricketGear Pro</div>
        <div class="sponsor-item">StatsHub Analytics</div>
        <div class="sponsor-item">SportsTech AI</div>
        <div class="sponsor-item">FastBowl Equipment</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Suggested Questions")
    
    # Batting Questions
    with st.expander("Batting Records", expanded=True):
        batting_questions = [
            "Most runs in ODI?",
            "Most runs in T20?",
            "Most runs in Test?",
            "Most centuries in ODI?",
            "Most sixes in T20?",
            "Most fours in T20?"
        ]
        for q in batting_questions:
            if st.button(q, key=f"bat_{q}"):
                st.session_state.suggested_query = q
    
    # Bowling Questions
    with st.expander("Bowling Records"):
        bowling_questions = [
            "Most wickets in ODI?",
            "Most wickets in T20?",
            "Most wickets in Test?",
        ]
        for q in bowling_questions:
            if st.button(q, key=f"bowl_{q}"):
                st.session_state.suggested_query = q
    
    # Career Stats
    with st.expander("Career Stats"):
        career_questions = [
            "Most matches in ODI?",
            "Most matches in Test?",
            "Most matches in T20?",
        ]
        for q in career_questions:
            if st.button(q, key=f"career_{q}"):
                st.session_state.suggested_query = q
    
    st.markdown("---")
    
    # Data info
    st.markdown("### Database Info")
    st.markdown("""
    <div class="info-box">
        <p style='margin: 5px 0; color: #37474f;'><b>Total Records:</b> 22,752</p>
        <p style='margin: 5px 0; color: #37474f;'><b>Batting:</b> 7,507 players</p>
        <p style='margin: 5px 0; color: #37474f;'><b>Bowling:</b> 7,638 players</p>
        <p style='margin: 5px 0; color: #37474f;'><b>Fielding:</b> 7,607 players</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tips
    st.markdown("### How to Use")
    st.markdown("""
    - Click suggested questions above
    - Type your own questions below
    - Mention format (ODI/T20/Test)
    - Ask about specific players
    """)
    
    st.markdown("---")
    
    # Sponsor section in sidebar
    st.markdown("### Our Partners")
    st.markdown("""
    <div style='background: white; padding: 0.8rem; border-radius: 8px; border: 2px solid #e3f2fd; text-align: center;'>
        <p style='margin: 5px 0; color: #1976d2; font-weight: 600; font-size: 0.85rem;'>CricketGear Pro</p>
        <p style='margin: 5px 0; color: #546e7a; font-size: 0.75rem;'>Premium Cricket Equipment</p>
    </div>
    <br>
    <div style='background: white; padding: 0.8rem; border-radius: 8px; border: 2px solid #e3f2fd; text-align: center;'>
        <p style='margin: 5px 0; color: #1976d2; font-weight: 600; font-size: 0.85rem;'>StatsHub Analytics</p>
        <p style='margin: 5px 0; color: #546e7a; font-size: 0.75rem;'>Advanced Sports Analytics</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize query engine
@st.cache_resource
def load_query_engine():
    return CricketQueryEngine()

@st.cache_resource
def load_vector_store():
    return get_vector_store()

try:
    query_engine = load_query_engine()
    vector_store = load_vector_store()
    
    if not vector_store:
        st.error("Data not found! Please run 'python scripts/build_index.py' first.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "**Welcome to Cricket Stats Bot!**\n\nI can help you with cricket statistics across ODI, T20, and Test formats.\n\n**Try asking:**\n- Who scored the most runs in ODI?\n- Most wickets in Test cricket?\n- Player statistics and records\n\nClick the suggested questions in the sidebar or type your own question below."
    }]

if "suggested_query" not in st.session_state:
    st.session_state.suggested_query = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle suggested query
if st.session_state.suggested_query:
    prompt = st.session_state.suggested_query
    st.session_state.suggested_query = None
else:
    prompt = st.chat_input("Ask me about cricket statistics...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Searching cricket database..."):
            try:
                # Try intelligent query engine first
                smart_answer = query_engine.answer_query(prompt)
                
                if smart_answer:
                    # Use smart answer
                    answer = smart_answer
                    message_placeholder.markdown(answer)
                    sources = []
                else:
                    # Fall back to vector search
                    qa_chain = get_qa_chain(vector_store)
                    response = qa_chain.invoke({"query": prompt})
                    answer = response['result']
                    sources = response.get('source_documents', [])
                    
                    message_placeholder.markdown(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(sources[:5], 1):
                                player = doc.metadata.get('player', 'Unknown')
                                format_type = doc.metadata.get('format', 'Unknown')
                                category = doc.metadata.get('category', 'Unknown')
                                
                                st.markdown(f"**{i}. {player}** - {format_type} {category}")
                                st.caption(doc.page_content[:150] + "...")
                                if i < len(sources[:5]):
                                    st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer with metrics
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size: 2rem;'>DATA</div>
        <div class="metric-value">22,752</div>
        <div class="metric-label">Total Players</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size: 2rem;'>FORM</div>
        <div class="metric-value">3</div>
        <div class="metric-label">Formats</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size: 2rem;'>CATS</div>
        <div class="metric-value">3</div>
        <div class="metric-label">Categories</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div style='font-size: 2rem;'>FAST</div>
        <div class="metric-value">&lt;2s</div>
        <div class="metric-label">Response Time</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #546e7a; font-size: 0.9rem;'>Built for cricket fans | Powered by AI and machine learning</p>", unsafe_allow_html=True)

