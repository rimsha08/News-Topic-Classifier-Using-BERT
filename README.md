# 📰 News Topic Classifier Using BERT

This is a web-based application built with **Streamlit** that classifies news headlines or short articles into one of four categories using a fine-tuned **BERT** model. The model is trained on the **AG News** dataset, which includes the categories: **World**, **Sports**, **Business**, and **Sci/Tech**.

---

## 🚀 Features

- Real-time news classification with BERT
- Interactive UI built using Streamlit
- Confidence scores displayed for each prediction
- Sidebar with collapsible model description and tech stack
- Emoji-enhanced labels for clarity and fun!

---

## 🧠 Model Description

The model is based on **bert-base-uncased** from HuggingFace Transformers. It has been fine-tuned on the AG News dataset consisting of 4 classes:

- 🌍 World  
- ⚽ Sports  
- 💼 Business  
- 🔬 Sci/Tech  

---

## 📁 Project Structure

News Topic Classifier Using BERT/
│
├── news-bert-model/ # Saved model directory
│ ├── config.json
│ ├── pytorch_model.bin
│ ├── tokenizer_config.json
│ └── vocab.txt
│
├── train.py # Fine-tuning script
├── evaluate.py # Evaluation script
├── app.py # Streamlit or Gradio app (for deployment)
├── requirements.txt # Dependencies
└── README.md # Project documentation

---
## 🛠️ Tech Stack

- **Model**: BERT (`bert-base-uncased`)
- **Dataset**: AG News (text classification)
- **Libraries**:
  - [Transformers (HuggingFace)](https://huggingface.co/transformers/)
  - [PyTorch](https://pytorch.org/)
  - [Streamlit](https://streamlit.io/)

## 🧪 Evaluation

| Metric     | Value |
|------------|-------|
| Accuracy   | 95%+  |
| F1-Score   | ~0.95 |

---

## 🛠 Setup Instructions
# Clone the repo
git clone https://github.com/your-username/news-topic-classifier-bert.git
cd news-topic-classifier-bert

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


