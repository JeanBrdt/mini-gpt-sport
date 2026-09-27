# ⚽ Mini-GPT from scratch: Football Commentator

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web-FF4B4B)

## 📌 Context
This project is a custom implementation of the Transformer architecture ("Attention Is All You Need", 2017) built entirely from scratch using PyTorch. 
Trained on a dataset of English Premier League play-by-play text, the model acts as a character-level autoregressive language model generating live football commentary.

## 🧠 Architectural Highlights (The "Senior" Touches)
Instead of just replicating the original 2017 paper, this implementation includes modern optimizations used in current LLMs (like LLaMA 3 / Mistral):
*   **RMSNorm (Root Mean Square Normalization):** Replaced standard LayerNorm for better computational efficiency without sacrificing performance.
*   **Multi-Head Causal Self-Attention:** Implemented the core routing mechanism with a causal mask (`torch.tril`) to prevent data leakage from future tokens.
*   **Character-Level Tokenization:** Custom encoder/decoder built from scratch to understand the raw mechanics of embedding spaces.
*   **Hardware Acceleration:** Native CUDA support for GPU training (`to(device)`).

## 🚀 Try it locally (Streamlit App)
An interactive UI is provided to test the model's generation capabilities.
```bash
pip install -r requirements.txt
streamlit run app.py