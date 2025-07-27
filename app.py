import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "news-bert-model"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to("cpu")
model.eval()


labels = ["World 🌍", "Sports ⚽", "Business 💼", "Sci/Tech 🔬"]

st.set_page_config(page_title="News Classifier", page_icon="📰", layout="centered")


with st.sidebar:
    st.title("📘 About This App")
    st.markdown("""
    This app uses a fine-tuned **BERT** model to classify news into 4 categories:
    - 🌍 World
    - ⚽ Sports
    - 💼 Business
    - 🔬 Sci/Tech

    **Use case:** Automatically tagging or organizing news content. 
                
    ---           
    """)

    st.subheader("🧠 Tech Stack")
    st.markdown("""
    - **Model:** BERT (bert-base-uncased)
    - **Dataset:** AG News (4 classes)
    - **Frameworks:** HuggingFace Transformers, PyTorch
    - **UI:** Streamlit
    """)

st.markdown(
    """
    <h1 style='text-align: left; color: #1f77b4; font-size: 38px;'>
        📰 AG News Classifier Using <span style='color: #d62728;'>BERT</span>
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Enter a news headline or short article below and get its category prediction:")

# User Input
text_input = st.text_area(
    "News Text",
    placeholder="e.g., Tesla announces breakthrough in self-driving technology",
    height=120
)


if st.button("🔍 Predict Category"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            # Tokenize
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to("cpu") for k, v in inputs.items()}  
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs).item()
                confidence = probs[0][pred_class].item()

        # Show result
        st.success(f"### 🎯 Prediction: {labels[pred_class]}")
        st.info(f"**Confidence Score:** `{confidence:.2%}`")


