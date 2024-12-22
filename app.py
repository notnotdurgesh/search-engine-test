import faiss
import pickle
import pandas as pd
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Configuration
DATA_PATH = Path("./data/misinformation_papers.csv")
INDEX_PATH = Path("models/faiss_index.pickle")
MODEL_NAME = "distilbert-base-nli-stsb-mean-tokens"

def initialize_faiss_index(embeddings, paper_ids):
    """Initialize and populate a new FAISS index."""
    embeddings = np.array([embedding for embedding in embeddings]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, paper_ids)
    return index

@st.cache_resource
def load_bert_model(name=MODEL_NAME):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)

@st.cache_data
def read_data(data_path=DATA_PATH):
    """Read the data from local CSV file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)

def vector_search(query, model, index, k=10):
    """Perform vector similarity search."""
    query_vector = model.encode(query)
    query_vector = np.array(query_vector).astype('float32')
    if len(query_vector.shape) == 1:
        query_vector = np.expand_dims(query_vector, axis=0)
    distances, indices = index.search(query_vector, k)
    return distances, indices

@st.cache_resource
def create_or_load_index(data, _model, load_existing=False, index_path=INDEX_PATH):
    """Create new FAISS index or load existing one."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    if load_existing and index_path.exists():
        try:
            with open(index_path, "rb") as h:
                index_data = pickle.load(h)
            return faiss.deserialize_index(index_data)
        except Exception as e:
            st.warning(f"Error loading existing index: {e}. Creating new index...")
    
    # Create new index
    embeddings = _model.encode(data.abstract.to_list(), show_progress_bar=True)
    index = initialize_faiss_index(embeddings, data.id.values)
    
    # Save the index
    try:
        with open(index_path, "wb") as h:
            pickle.dump(faiss.serialize_index(index), h)
    except Exception as e:
        st.warning(f"Failed to save index: {e}")
    
    return index

def display_paper_card(paper):
    """Display a single paper result in a card format."""
    with st.container():
        st.markdown(f"### {paper.original_title}")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Abstract**")
            st.write(paper.abstract)
        with col2:
            st.metric("Citations", paper.citations)
            st.metric("Year", paper.year)
        st.markdown("---")

def main():
    st.set_page_config(
        page_title="Research Paper Semantic Search",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Research Paper Semantic Search")
    st.write("Search through research papers using natural language queries.")

    try:
        # Load data and models
        data = read_data()
        model = load_bert_model()
        
        # Create or load FAISS index
        faiss_index = create_or_load_index(data, model, load_existing=True)

        # Sidebar filters
        st.sidebar.title("Search Filters")
        
        min_year = int(data.year.min())
        max_year = int(data.year.max())
        filter_year = st.sidebar.slider(
            "Publication year",
            min_year, max_year,
            (min_year, max_year)
        )
        
        max_citations = int(data.citations.max())
        filter_citations = st.sidebar.slider(
            "Minimum citations",
            0, max_citations, 0
        )
        
        num_results = st.sidebar.slider(
            "Number of results",
            5, 50, 10
        )

        # Search interface
        user_input = st.text_area(
            "Enter your search query",
            placeholder="Enter a research topic or description to find related papers...",
            height=100
        )

        if st.button("🔍 Search", type="primary") and user_input:
            with st.spinner("Searching..."):
                # Perform search
                D, I = vector_search([user_input], model, faiss_index, num_results)
                
                # Apply filters
                frame = data[
                    (data.year >= filter_year[0]) &
                    (data.year <= filter_year[1]) &
                    (data.citations >= filter_citations)
                ]
                
                # Display results
                results = [id_ for id_ in I.flatten() if id_ in set(frame.id)]
                
                if results:
                    st.success(f"Found {len(results)} matching papers")
                    for id_ in results:
                        display_paper_card(frame[frame.id == id_].iloc[0])
                else:
                    st.warning("No results found matching your filters.")

    except FileNotFoundError:
        st.error("Data file not found. Please ensure your CSV file is in the correct location.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

main()