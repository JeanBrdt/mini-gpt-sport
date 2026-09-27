import torch
import torch.nn as nn
from torch.nn import functional as F

n_embd = 32

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Le paramètre "gamma" (weight) que le réseau va apprendre
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calcul du carré de X, puis de la moyenne sur la dernière dimension
        # On ajoute epsilon pour éviter la division par zéro
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # On normalise et on multiplie par le poids appris
        return (x / rms) * self.weight


class Head(nn.Module):
    """ Une tête de Self-Attention """
    
    def __init__(self, head_size, block_size=8):
        super().__init__()
        # Nos 3 projections linéaires pour Q, K et V
        self.key = nn.Linear(head_size, head_size, bias=False)
        self.query = nn.Linear(head_size, head_size, bias=False)
        self.value = nn.Linear(head_size, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Calcul des scores d'attention ("affinités") : Q @ K^T / sqrt(d)
        # On transpose uniquement les 2 dernières dimensions de K (-2, -1)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)
        
        # Application du MASQUE CAUSAL : on remplace les zéros du haut par -infini
        tril = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        wei = wei.masked_fill(~tril, float('-inf'))
        
        # Softmax : Les -infini deviennent des probabilités de 0 (le futur est ignoré)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # Agrégation des valeurs V pondérées par l'attention
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
        
        return out


class MultiHeadAttention(nn.Module):
    """ Plusieurs têtes de Self-Attention exécutées en parallèle """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # On crée une liste contenant 'num_heads' têtes indépendantes
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # Une couche linéaire finale (W^O dans le papier original) 
        # pour mélanger les découvertes de toutes les têtes
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)

    def forward(self, x):
        B, T, C = x.shape
        assert C == self.num_heads * self.head_size
        # On découpe la dernière dimension en morceaux de taille head_size
        # puis on applique une tête sur chaque morceau
        x_splits = x.chunk(self.num_heads, dim=-1)
        out = torch.cat([h(xi) for h, xi in zip(self.heads, x_splits)], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ Un simple réseau linéaire suivi d'une non-linéarité """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # On multiplie la dimension par 4 (ex: de 32 vers 128)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # On recompresse vers la dimension d'origine (128 vers 32)
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        # x est de taille (B, T, n_embd)
        # La sortie reste de taille (B, T, n_embd)
        return self.net(x)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=32, block_size=256):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # On empile 3 blocs Transformer à la suite !
        self.blocks = nn.Sequential(
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            nn.LayerNorm(n_embd) # Une dernière normalisation avant la sortie
        )
        
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        if idx.size(1) > self.block_size:
            idx = idx[:, -self.block_size:]
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)    # On passe dans nos 3 blocs (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx est un tenseur (B, T) qui contient le contexte actuel
        for _ in range(max_new_tokens):
            if idx.size(1) > self.block_size:
                idx = idx[:, -self.block_size:]
            # 1. On récupère les prédictions (logits) via notre propre fonction forward
            logits, loss = self(idx)
            
            # 2. On ne s'intéresse qu'au TOUT DERNIER caractère prédit
            # logits devient (B, C) car on prend la dernière position du temps (T)
            logits = logits[:, -1, :]
            
            # 3. On applique un Softmax pour transformer les scores bruts en vraies probabilités (somme = 100%)
            probs = F.softmax(logits, dim=-1)
            
            # 4. On tire au sort le prochain caractère en respectant les probabilités
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # 5. On ajoute ce nouveau caractère à notre séquence existante
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx

class Block(nn.Module):
    """ Un bloc Transformer : communication (Attention) + calcul (FeedForward) """
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        
        # LayerNorm : Normalise les vecteurs pour stabiliser les mathématiques
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # On normalise -> On communique -> On ajoute au X d'origine (Résiduel)
        x = x + self.sa(self.ln1(x))
        # On normalise -> On calcule -> On ajoute au X d'origine (Résiduel)
        x = x + self.ffwd(self.ln2(x))
        return x