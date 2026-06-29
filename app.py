import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.express as px

# Page Config
st.set_page_config(page_title="NLP Similarity & Emotion App", layout="wide")

st.title("Text Similarity & Emotion Analysis")
st.markdown("This app uses free pretrained models to analyze text similarity and extract standard emotion scores without preprocessing.")

# --- Load Models (Cached for speed) ---
@st.cache_resource
def load_models():
    # Model 1: For Text Similarity
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Model 2: For Emotion Standard Scores (GoEmotions)
    emotion_model = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=1)
    return sim_model, emotion_model

sim_model, emotion_model = load_models()

# --- 1. Text input box ---
st.subheader("1. Enter Text")
default_text = "I am so excited to learn machine learning!\nThis error is making me very frustrated.\nI enjoy walking in the peaceful park.\nThe new robot design is completely neutral."
user_input = st.text_area("Enter sentences or words (one per line):", value=default_text, height=150)

texts = [t.strip() for t in user_input.split('\n') if t.strip()]

if len(texts) < 3:
    st.warning("Please enter at least 3 lines of text to generate meaningful graphs.")
else:
    # --- Strict Rule: No Preprocessing ---
    # Similarity Processing
    embeddings = sim_model.encode(texts)
    sim_matrix = cosine_similarity(embeddings)
    
    # Emotion Processing (GoEmotions)
    emotion_results = emotion_model(texts)
    
    # --- 2. GoEmotions Standard Scores ---
    st.subheader("2. Emotion Analysis (GoEmotions Standard Scores)")
    emo_data = []
    for text, result in zip(texts, emotion_results):
        emo_data.append({
            "Text": text, 
            "Dominant Emotion": result[0]['label'].capitalize(), 
            "Standard Score (Confidence)": result[0]['score']
        })
    df_emotions = pd.DataFrame(emo_data)
    st.dataframe(df_emotions.style.format({"Standard Score (Confidence)": "{:.4f}"}), use_container_width=True)

    # --- 3. Top Similarity Results ---
    st.subheader("3. Top Similarity Results")
    pairs = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            pairs.append({"Pair": f'"{texts[i]}" & "{texts[j]}"', "Score": sim_matrix[i][j]})
    
    df_pairs = pd.DataFrame(pairs).sort_values(by="Score", ascending=False).head(5)
    st.dataframe(df_pairs.style.format({"Score": "{:.4f}"}), use_container_width=True)

    # --- 4. Visualizations ---
    st.subheader("4. Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Similar Word/Sentence Pairs**")
        fig_bar = px.bar(df_pairs, x="Score", y="Pair", orientation='h', text_auto='.3f',
                         title="Top Pairwise Similarity Scores")
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("**Pairwise Similarity Heatmap**")
        fig_heat = px.imshow(sim_matrix, x=texts, y=texts, text_auto=".2f", aspect="auto",
                             color_continuous_scale="Viridis", title="Similarity Heatmap")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("**2D Embedding Plot (PCA)**")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    df_pca = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "Text": texts})
    
    fig_pca = px.scatter(df_pca, x="x", y="y", text="Text", size_max=60,
                         title="2D Plot of Text Embeddings")
    fig_pca.update_traces(textposition='top center')
    st.plotly_chart(fig_pca, use_container_width=True)

    # --- 5. Paul's Critical Thinking Standards ---
    st.subheader("5. Critical Thinking Notes (Paul's Standards)")
    st.markdown("""
    * **Clarity:** The input is raw text. The output provides both semantic similarity scores and categorical emotion standard scores.
    * **Accuracy:** Results are generated directly from HuggingFace's `all-MiniLM-L6-v2` and `roberta-base-go_emotions` models without any unsupported data manipulation.
    * **Precision:** Exact cosine similarity and emotion probability scores are displayed up to 4 decimal places.
    * **Relevance:** The extracted emotions directly correspond to the semantic tone, while the graphs strictly map the numeric similarity relationships.
    * **Logic:** Sentences with high emotion standard scores (e.g., > 0.8) indicate clear sentiment expression, which the model logically maps to specific GoEmotions categories.
    * **Significance:** Combining emotional context with semantic similarity provides a significantly deeper understanding of the text's true meaning.
    * **Fairness:** A limitation is that the emotion model might struggle with sarcasm, as it relies on literal text classification without external context.
    """)
