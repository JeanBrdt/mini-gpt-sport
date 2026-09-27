import streamlit as st
import torch
from model import BigramLanguageModel
from data import get_data

# --- Configuration de l'interface ---
st.set_page_config(page_title="Mini-GPT Sport", page_icon="⚽", layout="centered")
st.title("⚽ Mini-GPT : Football Commentator")
st.write("Ce modèle Transformer a été codé de zéro en PyTorch (avec RMSNorm et Multi-Head Attention) et entraîné sur des commentaires de football.")

# --- The Engineer Touch 1 : Le Cache ---
# @st.cache_resource évite de recharger le CSV et le modèle à chaque clic sur l'interface
@st.cache_resource
def load_system():
    # 1. On récupère notre vocabulaire et nos fonctions
    _, _, vocab_size, encode, decode = get_data()
    
    # 2. On détecte la RTX 2070
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 3. On initialise la structure vide (Attention: les hyperparamètres doivent être ceux du dernier entraînement !)
    model = BigramLanguageModel(vocab_size)
    
    # 4. On charge nos poids (notre "cerveau" sauvegardé)
    model.load_state_dict(torch.load('mini_gpt_sport.pth', map_location=device, weights_only=True))
    model = model.to(device)
    
    # --- The Engineer Touch 2 : Mode Inférence ---
    # model.eval() signale à PyTorch qu'on ne fait plus d'entraînement. 
    # Ça désactive le calcul des gradients et optimise la mémoire.
    model.eval() 
    
    return model, encode, decode, device

model, encode, decode, device = load_system()

# --- Zone Interactive ---
st.sidebar.header("Paramètres du modèle")
max_tokens = st.sidebar.slider("Longueur de la génération (caractères)", min_value=50, max_value=500, value=250)

prompt = st.text_input("Début de l'action :", value="90' - ")

if st.button("Générer la suite du match"):
    with st.spinner("Le modèle réfléchit..."):
        # On encode le texte de l'utilisateur
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        
        # On désactive complètement le moteur de gradients de PyTorch pour accélérer la génération
        with torch.no_grad():
            generated_tokens = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
            
        texte_genere = decode(generated_tokens)
        
        st.success(texte_genere)