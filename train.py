import torch
# On importe notre fonction depuis le fichier data.py qu'on a codé avant
from data import get_data

# 1. Chargement des données
train_data, val_data, vocab_size, encode, decode = get_data()

# 2. Nos premiers hyperparamètres
batch_size = 64 # B : Combien de séquences indépendantes on traite en parallèle
block_size = 256 # T : La longueur maximale de la séquence de contexte
max_iters = 5000

# 'cuda' désigne l'architecture de calcul des cartes NVIDIA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Calculs lancés sur l'accélérateur : {device}")

torch.manual_seed(1337) # Pour avoir toujours les mêmes résultats aléatoires

def get_batch(split):
    # On choisit le bon tenseur (entraînement ou validation)
    data = train_data if split == 'train' else val_data
    
    # On tire B nombres aléatoires qui seront les index de départ de nos séquences
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # X : Les séquences d'entrée de taille (B, T)
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # Y : Les séquences cibles, décalées d'un cran vers la droite (B, T)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # On déplace les tenseurs dans la VRAM de la RTX 2070
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')
print("Dimensions de la matrice X (Batch B, Time T) :", xb.shape)
print("Dimensions de la matrice Y (Batch B, Time T) :", yb.shape)

# Regardons l'intérieur du premier Batch
print("\n--- Exemple de ce que le modèle voit ---")
for b in range(batch_size): # Pour chaque séquence du batch
    for t in range(block_size): # Pour chaque pas de temps
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"Quand l'entrée est {context.tolist()} le modèle doit prédire {target}")
    break # On s'arrête après la première séquence pour ne pas inonder le terminal

# --- LA BOUCLE D'ENTRAÎNEMENT ---
from model import BigramLanguageModel

print("\n3. Initialisation du modèle...")
m = BigramLanguageModel(vocab_size)
m = m.to(device) # On déplace tout le réseau de neurones sur le GPU

# On crée notre optimiseur AdamW (Learning Rate de 1e-3, classique pour les petits modèles)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

print("\n4. Début de l'entraînement...")
for steps in range(max_iters):
    
    # 1. Échantillonner un nouveau batch
    xb, yb = get_batch('train')
    
    # 2. Forward pass : Calculer les prédictions et la loss
    logits, loss = m(xb, yb)
    
    # 3. Mettre à zéro les gradients de l'étape précédente
    optimizer.zero_grad(set_to_none=True)
    
    # 4. Backward pass : Calculer comment modifier chaque paramètre
    loss.backward()
    
    # 5. Mettre à jour les paramètres (Learning)
    optimizer.step()
    
    # Afficher la progression toutes les 100 étapes
    if steps % 100 == 0:
        print(f"Étape {steps} | Loss: {loss.item():.4f}")

print(f"Entraînement terminé ! Loss finale : {loss.item():.4f}")


print("\n--- TEST DE GÉNÉRATION ---")
# On commence avec le token 0 (qui est souvent un espace ou un saut de ligne)
# On crée un tenseur 2D de taille (1, 1) -> 1 Batch, 1 caractère de Time
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# On génère une séquence, on récupère le premier batch [0], et on convertit en liste Python
generated_tokens = m.generate(context, max_new_tokens=200)[0].tolist()

# On décode les nombres en texte
texte_genere = decode(generated_tokens)
print(texte_genere)

print("\n5. Sauvegarde du modèle...")
# On extrait le "dictionnaire d'état" (les poids appris) et on l'enregistre dans un fichier
torch.save(m.state_dict(), 'mini_gpt_sport.pth')
print("✅ Modèle sauvegardé avec succès ! Prêt pour le déploiement Web.")