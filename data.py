from datasets import load_dataset

def get_data():
    print("1. Téléchargement du dataset Hugging Face...")
    # On charge uniquement la partie 'train' du dataset
    dataset = load_dataset("falahmanalodi/Football-Commentary", split="train")
    
    print("2. Concaténation du texte...")
    # Le dataset est un dictionnaire géant. La colonne qui nous intéresse s'appelle 'Comment'.
    # On prend toutes les phrases et on les colle ensemble avec un espace.
    commentary = " ".join(dataset['Comment'])
    print(f"-> Longueur totale du corpus : {len(commentary):,} caractères.")

    print("3. Création du vocabulaire...")
    # set(commentary) : prend tous les caractères uniques (enlève les doublons)
    # list(...) : convertit le set en liste pour pouvoir la trier
    # sorted(...) : trie la liste par ordre alphabétique
    chars = sorted(list(set(commentary)))
    vocab_size = len(chars)
    
    print(f"-> Taille du vocabulaire (vocab_size) : {vocab_size} caractères uniques.")
    print(f"-> Voici notre alphabet complet :\n{''.join(chars)}")

    # 4. Création de l'Encodeur / Décodeur
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 5. Conversion en Tenseur PyTorch
    import torch
    data = torch.tensor(encode(commentary), dtype=torch.long)

    # 6. Séparation Train / Validation (90% / 10%)
    n = int(0.9 * len(data)) 
    train_data = data[:n]
    val_data = data[n:]

    # ON RENVOIE BIEN LES 5 VALEURS ATTENDUES PAR TRAIN.PY
    return train_data, val_data, vocab_size, encode, decode

    

# Ce bloc s'exécute uniquement si tu lances directement "python data.py"
if __name__ == "__main__":
    train_data, val_data, vocab_size, encode, decode = get_data()
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")
    print(f"Vocabulary size: {vocab_size}")