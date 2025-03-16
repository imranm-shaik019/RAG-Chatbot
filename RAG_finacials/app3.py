import os  # Provides functions for interacting with the operating system (e.g., file paths)
import streamlit as st  # Streamlit is used for creating web applications and UI components
import faiss  # FAISS (Facebook AI Similarity Search) is used for fast vector search and indexing
import json  # JSON module for reading and writing JSON data files
import numpy as np  # NumPy for numerical operations, often used for handling embeddings
from sentence_transformers import SentenceTransformer  # Pretrained transformer model for generating sentence embeddings
from rank_bm25 import BM25Okapi  # BM25Okapi for ranking documents using the BM25 retrieval algorithm
from transformers import pipeline  # Transformers pipeline for various NLP tasks like text generation, summarization, etc.
import streamlit as st  # Streamlit is used for UI feedback


#### 1. Data Collection & Preprocessing ####

# Define paths for data files
DATA_DIR = "data"  # Directory where data files are stored
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_pdfs.bin")  # Path to FAISS index file for vector search
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "pdf_chunks.json")  # Path to store extracted PDF chunks metadata
DOC_JSON_PATH = os.path.join(DATA_DIR, "all_docs.json")  # Path to document data used for BM25 retrieval

# Load FAISS index for vector search
try:
    # Check if the FAISS index file exists before attempting to load
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index file {FAISS_INDEX_PATH} not found!")
    
    # Read the FAISS index from the specified path
    faiss_pdf = faiss.read_index(FAISS_INDEX_PATH)

except Exception as e:
    # Display error message in the sidebar if the index file cannot be loaded
    st.sidebar.error(f"Error loading FAISS index: {e}")
    
    # Set FAISS index to None to prevent errors in downstream processing
    faiss_pdf = None


# Load PDF chunk metadata from the JSON file
try:
    # Open and read the JSON file containing metadata of extracted PDF chunks
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)  # Load JSON data into a Python dictionary
except Exception as e:
    # Display an error message in the Streamlit sidebar if loading fails
    st.sidebar.error(f"Error loading chunk metadata: {e}")
    
    # Set metadata to None to prevent issues in downstream processing
    chunk_metadata = None


# Load BM25 model for keyword-based retrieval
try:
    # Open and read the JSON file containing document data
    with open(DOC_JSON_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)  # Load document data into a Python list
    
    # Preprocess documents for BM25 by converting text to lowercase and tokenizing into words
    bm25_corpus = [doc["text"].lower().split() for doc in documents]  
    
    # Initialize BM25 model using the preprocessed corpus
    bm25 = BM25Okapi(bm25_corpus)
except Exception as e:
    # Display an error message in the Streamlit sidebar if loading fails
    st.sidebar.error(f"Error loading BM25 data: {e}")
    
    # Set BM25 model to None to prevent errors in downstream processing
    bm25 = None


# === Financial Query Classification ===

# Initialize a zero-shot text classification pipeline using the BART-Large-MNLI model.
# This model is pre-trained for natural language inference (NLI) and can classify text
# into given categories without requiring retraining.
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def is_financial_query(query: str) -> bool:
    """
    Determines if the given query is related to finance using zero-shot classification.

    Args:
        query (str): The user's query to classify.

    Returns:
        bool: True if the query is classified as financial, otherwise False.
    """
    # Define candidate labels for classification
    candidate_labels = ["financial", "non-financial"]

    # Use the classifier to determine which label best matches the query
    result = classifier(query, candidate_labels)

    # Return True if the top predicted label is "financial", otherwise False
    return result['labels'][0] == 'financial'


#### 2. Basic RAG Implementation ####

# Load SentenceTransformer for generating text embeddings
try:
    # Initialize the SentenceTransformer model with a pre-trained embedding model.
    # "all-MiniLM-L6-v2" is a lightweight transformer-based model optimized for sentence embeddings.
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    # Display an error message in the Streamlit sidebar if model loading fails
    st.sidebar.error(f"Error loading embedder: {e}")
    
    # Set embedder to None to prevent issues in downstream processing
    embedder = None


# === Basic Retrieval ===
def basic_retrieve(query: str):
    """
    Retrieves relevant document chunks based on the query using FAISS for vector search.
    
    Args:
        query (str): The user's query to retrieve results for.
    
    Returns:
        list: A list of dictionaries containing retrieval results, including PDF file, chunk ID,
              similarity score, and text. Returns an empty list if an error occurs or models are not loaded.
    """
    if not embedder or not faiss_pdf or not chunk_metadata:
        return []  # Return empty list if required models or data are not loaded
    
    try:
        # Encode the query into an embedding
        query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search the FAISS index for the top 3 most similar chunks
        distances, indices = faiss_pdf.search(query_emb, 3)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunk_metadata):  # Ensure the index is within bounds
                # Calculate similarity score (inverse of distance)
                similarity_score = 1 / (1 + distances[0][i]) if distances[0][i] > 0 else 1.0
                
                # Append result with metadata and similarity score
                results.append({
                    "pdf_file": chunk_metadata[idx]["pdf_file"],  # PDF file name
                    "chunk_id": idx,  # Chunk ID
                    "similarity": round(similarity_score, 4),  # Similarity score rounded to 4 decimal places
                    "text": chunk_metadata[idx]["text"]  # Text content of the chunk
                })

        # Debugging: Uncomment to log retrieved results
        # st.write("DEBUG: Retrieved results →", results)
        
        return results
    except Exception as e:
        st.error(f"Error in basic retrieval: {e}")
        return []  # Return empty list if an error occurs

#### 3. Advanced RAG Implementation ####
# === Multi-Stage Retrieval ===
def multi_stage_retrieve(query: str):
    """
    Performs multi-stage document retrieval using BM25 and FAISS.

    Args:
        query (str): The user's search query.

    Returns:
        list: A list of top retrieved document chunks with metadata.
    """

    # Ensure that all required components are available before proceeding
    if not bm25 or not chunk_metadata or not embedder or not faiss_pdf:
        return []

    try:
        # === Stage 1: Keyword-Based Retrieval (BM25) ===
        # Compute BM25 scores for the query by tokenizing and matching against the corpus
        bm25_scores = bm25.get_scores(query.lower().split())

        # Get the top 5 document indices with the highest BM25 scores
        top_doc_indices = np.argsort(bm25_scores)[-5:][::-1]

        # Filter out any indices that exceed the available metadata length
        filtered_indices = [idx for idx in top_doc_indices if idx < len(chunk_metadata)]

        # === Stage 2: Semantic Similarity with Sentence Embeddings (FAISS) ===
        # Generate an embedding for the query using the SentenceTransformer model
        query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).squeeze()

        # Encode the text of the top retrieved BM25 documents into embeddings
        filtered_embeddings = np.array([
            embedder.encode(chunk_metadata[idx]["text"], convert_to_numpy=True, normalize_embeddings=True)
            for idx in filtered_indices
        ])

        # Compute cosine similarity between query embedding and retrieved document embeddings
        similarities = np.dot(filtered_embeddings, query_emb)

        # Get the top 3 most relevant document indices based on similarity scores
        sorted_indices = np.argsort(similarities)[-3:][::-1]

        # Format the retrieval results with metadata
        results = [
            {
                "pdf_file": chunk_metadata[filtered_indices[idx]]["pdf_file"],  # PDF file name
                "chunk_id": filtered_indices[idx],  # Chunk index in the document
                "similarity": round(similarities[idx], 4),  # Similarity score (rounded)
                "text": chunk_metadata[filtered_indices[idx]]["text"]  # Retrieved text snippet
            }
            for idx in sorted_indices
        ]

        # Debugging: Uncomment to inspect retrieved results
        # st.write("DEBUG: Retrieved results →", results)  

        return results

    except Exception as e:
        # Display an error message in the Streamlit app if retrieval fails
        st.error(f"Error in multi-stage retrieval: {e}")
        return []


#### 5. Guard Rail Implementation and Response Generation ####
# === Response Generation ===
def generate_response(query: str, mode: str = "multi-stage") -> str:
    """
    Generates a response based on retrieved document chunks.

    Args:
        query (str): The user's search query.
        mode (str, optional): Retrieval mode; either "multi-stage" (default) or "basic".

    Returns:
        str: A response based on the retrieved documents, or an error message if retrieval fails.
    """

    # === Step 1: Ensure All Necessary Components Are Loaded ===
    # Check if all required models and data are available
    if not all([embedder, faiss_pdf, chunk_metadata, bm25]):
        return "One or more models or data failed to load. Please check the sidebar for errors."

    # === Step 2: Query Classification (Financial or Non-Financial) ===
    # Use zero-shot classification to determine if the query is finance-related
    if not is_financial_query(query):
        return "This is not a financial query. Please ask something related to finance."

    # === Step 3: Retrieve Relevant Documents ===
    # Perform retrieval using the selected mode:
    # - "multi-stage" (BM25 + FAISS for hybrid search)
    # - "basic" (BM25 only)
    results = multi_stage_retrieve(query) if mode == "multi-stage" else basic_retrieve(query)
    
    # Check if any relevant results were found
    if not results:
        return "No relevant data found."

    # Select the top-ranked result (most relevant match)
    top_result = results[0]  
    similarity_score = top_result.get("similarity", 0.0)  # Default to 0.0 if similarity is missing

    # === Step 4: Confidence-Based Response Handling ===
    # If the similarity score is too low (<= 0.5), indicate uncertainty in the response
    if similarity_score <= 0.5:
        return f"**Answer:** Sorry! I cannot answer this query at the moment.\n\n**Confidence Score:** {similarity_score:.4f}"    

    # === Step 5: Return the Retrieved Response ===
    return f"**Answer:** {top_result['text']}\n\n**Confidence Score:** {similarity_score:.4f}"


#### 4. UI Development (e.g., Streamlit) and 6.Testing & Validation ####
# === Streamlit UI ===
def main():
    """
    Streamlit UI for financial chatbot.
    Users enter queries, which are processed upon pressing Enter.
    """
    st.title("RAG Financial Chatbot")

    retrieval_mode = st.sidebar.selectbox("Retrieval Mode", ["multi-stage", "basic"])
    user_query = st.text_area("Enter your query:", key="query", height=100)  # Larger input box

    # Automatically process query when user presses Enter
    if user_query.strip():
        with st.spinner("Processing..."):
            answer = generate_response(user_query, mode=retrieval_mode)
            
            # Split the answer into "Answer" and "Confidence Score"
            if isinstance(answer, str) and "**Answer:**" in answer and "**Confidence Score:**" in answer:
                answer_text = answer.split("**Answer:**")[1].split("**Confidence Score:**")[0].strip()
                confidence_score = answer.split("**Confidence Score:**")[1].strip()
            else:
                answer_text = answer
                confidence_score = "N/A"

            # Display results sequentially
            st.markdown("### Answer")
            st.info(answer_text)  # Display answer in a box

            st.markdown("### Confidence Score")
            st.success(confidence_score)  # Display confidence score in a box below the answer

if __name__ == "__main__":
    main()
