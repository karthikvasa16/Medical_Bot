import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# Configure Gemini API (replace with your key)
st.sidebar.header("🔐 Gemini API Key")
api_key = st.sidebar.text_input("Paste your Gemini API Key:", type="password")
genai.configure(api_key=api_key)

# Load dataset and embeddings
df = pd.read_csv("combined_output.csv")
embeddings = pd.read_csv("symptom_embeddings.csv").values.astype(np.float32)  # Load CSV as numpy array

# Load embedding model (for query only)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("🧠 Medical RAG Bot using Gemini Flash 2.5")
user_input = st.text_input("Enter your symptoms:")

if user_input:
    # Embed user query
    query_embedding = embedder.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarity
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_idx = int(np.argmax(cos_scores))
    top_score = float(cos_scores[top_idx])  # Extract top similarity score

    # Get matched record
    row = df.iloc[top_idx]
    prompt = f"""
    You are a helpful medical assistant. Based on the following symptom: "{user_input}",
    provide a detailed medical diagnosis based on the match:

    Disease: {row['disease']}
    Severity: {row['severity']}
    Description: {row['Description']}

    Format your response clearly and informatively.
    """

    # Query Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Display
    st.markdown("### 🩺 Predicted Diagnosis:")
    st.write(response.text)
    st.markdown(f"**🔍 Similarity Score:** `{top_score:.4f}`")
