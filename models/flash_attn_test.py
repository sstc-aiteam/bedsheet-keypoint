import torch
from flash_attn import flash_attn_func

# Parameters
B = 2                 # batch size
N = 1024              # sequence length (e.g., H * W for images)
C = 256               # embed dim (must be divisible by num_heads)
num_heads = 8
head_dim = C // num_heads
dropout_p = 0.0
causal = False

# Make Q, K, V (shape: [B, N, num_heads, head_dim])
q = torch.randn(B, N, num_heads, head_dim, device="cuda", dtype=torch.float16)
k = torch.randn(B, N, num_heads, head_dim, device="cuda", dtype=torch.float16)
v = torch.randn(B, N, num_heads, head_dim, device="cuda", dtype=torch.float16)

# Other optional arguments (use None or reasonable defaults if unsure)
window_size = None
softcap = None
alibi_slopes = None
deterministic = False

# Call flash_attn_func
out, lse= flash_attn_func(
    q,
    k,
    v,
    dropout_p
)

print("Output shape:", out.shape)          # (B, N, num_heads, head_dim)
print("Logsumexp shape:", lse.shape)       # (B, N, num_heads)
