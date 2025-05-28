import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# Configure Gemini API (replace with your key)
genai.configure(api_key="AIzaSyDexffYjmTQRUfLPtfkd65yrCXRgYr0S9c")

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
    top_index = similarities.argmax()
    matched = df.iloc[top_index]
    rag_context = f"""
Disease: {matched['disease']}
Symptoms: {matched['common_symptom']}
Severity (1–5): {matched['severity']}
Description: {matched['Description']}
"""
    prompt = f"""
You are a medical assistant. Use the following context to answer a patient's query.

Context:
{rag_context}

The patient reports: {user_input}

Based on this, answer the following:
1. What is the most likely disease?
2. How severe is it?
3. Give a brief description.
4. What are the recommended treatments and precautions?
5. What advice would you give the patient?
"""

    # Query Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # Display
    st.markdown("### 🩺 Predicted Diagnosis:")
    st.write(response.text)
    st.markdown(f"**🔍 Similarity Score:** `{top_score:.4f}`")
