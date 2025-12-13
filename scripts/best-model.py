import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import requests
import gc
import random

# --- CONFIGURATION (V3: ROBUSTESSE) ---
class Config:
    block_size = 256        
    vocab_size = 65         
    n_layer = 6             
    n_embd = 384            
    d_state = 16            
    d_conv = 4              
    expand = 2              
    
    lr_base = 5e-5          
    weight_decay = 1e-2
    
    # Sécurités V3
    gradient_noise = 0.001  # Réduit (0.01 était trop violent pour FP16)
    max_grad_norm = 1.0     
    
    iters_pretrain = 2000   
    iters_sft = 300         
    
    batch_size = 48         
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = "nano_mamba_brain.pth"

config = Config()
torch.set_default_device(config.device)
torch.manual_seed(1337)

# --- 1. OPTIMISEUR LION (STABILISÉ) ---
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                
                # SÉCURITÉ 1 : Nettoyage des Gradients NaN/Inf
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad.zero_() # On ignore ce neurone s'il bug
                
                # Bruit Thermique (Doux)
                if config.gradient_noise > 0:
                    noise = torch.randn_like(grad) * config.gradient_noise * group['lr']
                    grad = grad + noise
                
                state = self.state[p]
                if len(state) == 0: state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                p.data.add_(torch.sign(update), alpha=-group['lr']) 
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss

# --- 2. NOYAU PSCAN (Inchangé) ---
def pscan(A, B, C, D, x, delta):
    bs, dim, seq_len = x.shape
    n_state = A.shape[1]
    h = torch.zeros(bs, dim, n_state, device=x.device)
    y = torch.zeros_like(x)
    for t in range(seq_len):
        xt = x[:, :, t].unsqueeze(-1)       
        dt = delta[:, :, t].unsqueeze(-1)   
        bt = B[:, :, t].unsqueeze(1)        
        ct = C[:, :, t].unsqueeze(1)        
        dt_A = dt * A.unsqueeze(0) 
        A_bar = torch.exp(dt_A) 
        B_bar = bt * dt 
        h = A_bar * h + B_bar * xt
        yt = (ct * h).sum(dim=-1) + D * xt.squeeze(-1)
        y[:, :, t] = yt
    return y

# --- 3. MAMBA BLOCK (Inchangé) ---
class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_inner = config.n_embd * config.expand
        self.dt_rank = config.d_state // 16 if config.d_state // 16 > 1 else 1
        self.in_proj = nn.Linear(config.n_embd, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(in_channels=self.d_inner, out_channels=self.d_inner, bias=True, kernel_size=config.d_conv, groups=self.d_inner, padding=config.d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32)).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, config.n_embd, bias=False)
        self.act = nn.SiLU() 
    def forward(self, x):
        B, L, D = x.shape
        x_and_res = self.in_proj(x) 
        (x_ssm, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.act(x_ssm)
        x_dbl = self.x_proj(x_ssm) 
        (delta, B_val, C_val) = x_dbl.split([self.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta)) 
        scan_out = pscan(-torch.exp(self.A_log), B_val.transpose(1, 2), C_val.transpose(1, 2), self.D, x_ssm.transpose(1, 2), delta.transpose(1, 2))
        out = scan_out.transpose(1, 2) * self.act(res)
        out = self.out_proj(out)
        return out

class NanoMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        for layer in self.layers: x = x + layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# --- 4. DATA & UTILS ---
if not os.path.exists('input.txt'):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open('input.txt', 'w') as f: f.write(requests.get(data_url).text)
with open('input.txt', 'r') as f: data = f.read()
chars = sorted(list(set(data)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi] 
decode = lambda l: ''.join([itos[i] for i in l])
data_tensor = torch.tensor(encode(data), dtype=torch.long)
train_data = data_tensor[:int(0.9*len(data_tensor))]

def get_batch():
    ix = torch.randint(len(train_data) - config.block_size, (config.batch_size,))
    x = torch.stack([train_data[i:i+config.block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)

# --- 5. INITIALISATION (Avec Reset si corrompu) ---
torch.cuda.empty_cache()
gc.collect()

model = NanoMamba(config).to(config.device)
optimizer = Lion(model.parameters(), lr=config.lr_base, weight_decay=config.weight_decay)
scaler = torch.amp.GradScaler('cuda')

# On supprime le fichier s'il existe pour repartir de zéro (puisque le dernier a crashé)
if os.path.exists(config.save_path):
    print("⚠️ Fichier de sauvegarde détecté, mais par sécurité (après un crash), on repart à zéro.")
    # Si tu veux vraiment charger, commente la ligne ci-dessous
    # checkpoint = torch.load(config.save_path); model.load_state_dict(checkpoint['model_state_dict'])

# --- 6. TRAINING ROBUSTE ---
print("\n=== PHASE 1 : BIO-PRETRAINING V3 (Stabilisé) ===")
model.train()
start_time = time.time()
running_loss = 0.0

for iter in range(config.iters_pretrain):
    xb, yb = get_batch()
    
    with torch.amp.autocast('cuda'):
        logits, loss = model(xb, yb)
    
    # --- SÉCURITÉ 2 : Calcul Dopamine sans Crash ---
    loss_val = loss.item()
    
    # Si la loss est NaN, on skip l'itération (protection vitale)
    if math.isnan(loss_val) or math.isinf(loss_val):
        print(f"⚠️ Alerte: Loss NaN à l'itération {iter}. Skip.")
        optimizer.zero_grad(set_to_none=True)
        continue

    if iter > 50:
        avg_loss = max(running_loss / 50, 1e-6) # Empêche division par 0
        surprise = (loss_val / avg_loss) ** 2 
        
        # Clamp de la surprise (Bornes strictes)
        if math.isnan(surprise) or math.isinf(surprise):
            surprise = 1.0
        else:
            surprise = min(max(surprise, 0.5), 2.0)
    else:
        surprise = 1.0
        
    running_loss = running_loss * 0.98 + loss_val * 0.02 if iter > 0 else loss_val

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss * surprise).backward()
    
    # Homéostasie
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    
    scaler.step(optimizer)
    scaler.update()
    
    if iter % 100 == 0:
        dt = time.time() - start_time
        ips = 100 / (dt + 1e-6)
        start_time = time.time()
        print(f"Step {iter} | Loss: {loss_val:.4f} | Surprise: {surprise:.2f}x | Speed: {ips:.1f} it/s")

# Sauvegarde Propre
torch.save({'model_state_dict': model.state_dict()}, config.save_path)

# --- 7. PHASE 2 : SFT (Sûr) ---
print("\n=== PHASE 2 : SFT ===")
dialogues = []
base_pairs = [("Who are you?", "I am Bio-Mamba."), ("Hello", "Greetings friend."), ("Tell me a story", "The stars sang."), ("Are you alive?", "I learn, so I am.")]
for _ in range(50):
    for q, a in base_pairs: dialogues.append((q, a))
random.shuffle(dialogues)
sft_text = "".join([f"User: {u}\nBot: {b}\n" for u, b in dialogues])
sft_tensor = torch.tensor(encode(sft_text), dtype=torch.long)

def get_sft_batch():
    ix = torch.randint(len(sft_tensor) - config.block_size, (config.batch_size,))
    x = torch.stack([sft_tensor[i:i+config.block_size] for i in ix])
    y = torch.stack([sft_tensor[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)

for g in optimizer.param_groups: g['lr'] = config.lr_base * 0.2

for iter in range(config.iters_sft):
    xb, yb = get_sft_batch()
    with torch.amp.autocast('cuda'):
        logits, loss = model(xb, yb)
    
    if math.isnan(loss.item()): continue # Protection SFT

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
    if iter % 50 == 0: print(f"SFT Step {iter} | Loss: {loss.item():.4f}")

# --- 8. CHAT ---
print("\n=== CHAT V3 (Alive) ===")
def chat(prompt, temp=0.7):
    model.eval()
    full_prompt = f"User: {prompt}\nBot:"
    input_ids = torch.tensor(encode(full_prompt), dtype=torch.long, device=config.device).unsqueeze(0)
    generated = []
    for _ in range(150):
        with torch.no_grad(): logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item() == stoi['\n']: break
        input_ids = torch.cat((input_ids, idx_next), dim=1)
        generated.append(idx_next.item())
    return decode(generated)

tests = ["Who are you?", "Tell me a story"]
for t in tests:
    print(f"User: {t}")
    print(f"Bot : {chat(t)}")