import torch
import torch.nn as nn
import torch.nn.functional as F

# Actual run configs
# BLOCK_SIZE = 256
# EMBEDDING_DIM = 384
# NUM_HEAD = 6
# NUM_TRANSFORMER_BLOCK = 6
# DROPOUT = 0.2
# VOCAB_SIZE = 65

# Test run configs
BLOCK_SIZE = 3
EMBEDDING_DIM = 10
NUM_HEAD = 2
NUM_TRANSFORMER_BLOCK = 1
DROPOUT = 0.2
VOCAB_SIZE = 10

class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.char_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, device='cuda')
        self.pos_embedding = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM, device='cuda')

    def forward(self, x):
        _, T = x.shape

        tok_emb = self.char_embedding(x) # (B,T,C)
        pos_emb = self.pos_embedding(torch.arange(T, device='cuda')) # (T,C)
        x_emb = tok_emb + pos_emb # (B,T,C)

        return x_emb

class AttentionLayer(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False, device='cuda')
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False, device='cuda')
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False, device='cuda')
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE, device='cuda')))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
  
        # wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionLayer(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, EMBEDDING_DIM, device='cuda')
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        head_attention_outs = []
        for head in self.heads:
            attn_out = head(x)
            head_attention_outs.append(attn_out)

        out = torch.cat(head_attention_outs, dim=-1)
        out = self.proj(out)
        # out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ff_linear_1 = nn.Linear(n_embd, 4 * n_embd, device='cuda')
        self.ff_act_fn = nn.ReLU()
        self.ff_linear_2 = nn.Linear(4 *n_embd, n_embd, device='cuda')
        
        self.ff_dropout = nn.Dropout(DROPOUT)
        self.ff_layer_norm = nn.LayerNorm(n_embd, device='cuda')

    def forward(self, x):
        ff_lin_1_out = self.ff_linear_1(self.ff_layer_norm(x))
        ff_act_out = self.ff_act_fn(ff_lin_1_out)
        ff_lin_2_out = self.ff_linear_2(ff_act_out)
        # ff_with_residual_out = x + self.ff_dropout(ff_lin_2_out)
        ff_with_residual_out = x + ff_lin_2_out

        return ff_with_residual_out

class TransformerBLock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.feedforward = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd, device='cuda')
        self.ln2 = nn.LayerNorm(n_embd, device='cuda')

    def forward(self, attention_with_residual_out):
        attention_with_residual_out = attention_with_residual_out + self.sa(self.ln1(attention_with_residual_out))
        ff_output = self.feedforward(attention_with_residual_out)

        return ff_output

class TorchTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embeddings = Embeddings()
        self.blocks = nn.Sequential(*[TransformerBLock(EMBEDDING_DIM, n_head=NUM_HEAD) for _ in range(NUM_TRANSFORMER_BLOCK)])
        self.ln_f = nn.LayerNorm(EMBEDDING_DIM, device='cuda') # final layer norm
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE, device='cuda')

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        x_emb = self.embeddings(idx)
        x = self.blocks(x_emb) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return x_emb, loss
