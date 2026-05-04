# Course 4: Deep Learning with PyTorch

**Prerequisites:** Course 1 (Python for ML Engineers), Course 3 (Audio ML with PyTorch — for tensor shape intuition).
**Goal:** After this course you can read every file in `src/tts_pipeline/models/` and `src/tts_pipeline/training/` completely, trace any tensor from input to loss, and explain every design decision.

---

## Module 1 — Tensors: The Fundamental Data Structure

### 1.1 What a tensor is

A tensor is an N-dimensional array with:
- A **dtype** — the element type (float32, int64, bool, ...)
- A **device** — where the data lives (CPU RAM or GPU VRAM)
- A **shape** — the size of each dimension
- A **storage** — the flat block of memory the data actually occupies

```python
import torch

t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t.dtype)    # torch.float32
print(t.device)   # device(type='cpu')
print(t.shape)    # torch.Size([2, 2])
print(t.ndim)     # 2
```

### 1.2 Creating tensors

```python
# From data
torch.tensor([1.0, 2.0, 3.0])              # float32 by default for floats

# Filled
torch.zeros(3, 4)                           # 3×4 of 0.0
torch.ones(2, 80, 256)                      # 2×80×256 of 1.0
torch.full((2, 5), fill_value=-1.0)         # 2×5 of -1.0

# Random
torch.rand(2, 80, 256)                      # uniform [0, 1)
torch.randn(2, 80, 256)                     # standard normal N(0,1)
torch.randint(0, 20, (2, 8))                # integers in [0, 20)

# Like another tensor (same shape, device, dtype)
x = torch.randn(2, 80, 256)
torch.zeros_like(x)                         # same shape as x, filled with 0
torch.randn_like(x)                         # same shape as x, random normal

# Sequences
torch.arange(0, 10, step=2)                 # [0, 2, 4, 6, 8]
torch.linspace(0.0, 1.0, steps=5)           # [0.0, 0.25, 0.5, 0.75, 1.0]
```

**In this project:** [model.py:78-80](../src/tts_pipeline/models/matcha/model.py#L78-L80)
```python
t = torch.rand(B, device=phoneme_ids.device)        # uniform time samples
x0 = torch.randn_like(mel_targets)                  # noise same shape as mel
x_t, u_t = self.cfm.sample_xt(x0, mel_targets, t)
```

### 1.3 The batch dimension convention

Every tensor in training has a **batch dimension** as dimension 0. This is universally `B` (batch size):

```
phoneme_ids:   (B, T_text)          — B sequences of T_text tokens
mel_targets:   (B, n_mels, T_mel)   — B mel spectrograms
phoneme_lengths: (B,)               — length of each sequence in the batch
loss:          ()                   — scalar (no dimensions, just a number)
```

Batching is what lets GPUs process many examples in parallel. A single forward pass with B=16 is roughly the same wall-clock time as B=1.

### 1.4 Broadcasting

Broadcasting allows operations between tensors of different shapes if they are compatible:

```python
# Rule: align shapes from the right, fill missing dims with 1
a = torch.randn(2, 80, 256)   # (B, n_mels, T)
b = torch.randn(2, 1, 1)      # (B, 1, 1)
(a + b).shape                  # (2, 80, 256) — b is broadcast across mel and time
```

**In this project:** [flow.py:18-20](../src/tts_pipeline/models/matcha/flow.py#L18-L20)
```python
t_b = t[:, None, None]   # (B,) → (B, 1, 1) — adds two dims for broadcasting over (B, n_mels, T)
mu_t = t_b * x1 + (1 - t_b) * x0   # works because t_b broadcasts over n_mels and T
```

The `None` index inserts a size-1 dimension. `t[:, None, None]` turns `(B,)` into `(B, 1, 1)`, which then broadcasts to match `(B, n_mels, T_mel)`.

### 1.5 Moving between devices

```python
model = MyModel()
model.to("cuda")           # move all parameters to GPU

x = torch.randn(2, 80, 256)
x = x.to("cuda")           # move tensor to GPU

x.device                   # device(type='cuda', index=0)
x.cpu()                    # move back to CPU (returns new tensor)
```

**Critical rule:** model parameters and input tensors must be on the same device. If they're not, you get a `RuntimeError` immediately.

```python
# This crashes:
model.to("cuda")
x = torch.randn(2, 80, 256)   # on CPU
model(x)                        # RuntimeError: expected CUDA tensor, got CPU tensor
```

**In this project:** [model.py:108](../src/tts_pipeline/models/matcha/model.py#L108)
```python
x0 = torch.randn(..., device=phoneme_ids.device)   # always match device to input
```

---

## Module 2 — Autograd: Automatic Differentiation

### 2.1 What autograd does

Deep learning training requires computing gradients — the derivative of the loss with respect to every parameter. Doing this by hand for a 50-layer network is impossible. PyTorch autograd does it automatically.

The key insight: PyTorch records every operation applied to a tensor that has `requires_grad=True`. This record is the **computation graph**. When you call `.backward()`, it traverses the graph in reverse and accumulates gradients.

### 2.2 The gradient flow

```python
# Minimal training loop — naked (without Lightning)
x = torch.randn(2, 80, 30)        # input (no grad needed)
target = torch.randn(2, 80, 30)   # target (no grad needed)

model = nn.Linear(80, 80)         # has parameters with requires_grad=True by default

pred = model(x)                    # forward pass — graph is built
loss = ((pred - target) ** 2).mean()

# At this point: loss has a grad_fn (part of the graph)
print(loss.requires_grad)          # True
print(loss.grad_fn)                # MeanBackward0

loss.backward()                    # compute d(loss)/d(param) for every param

for name, param in model.named_parameters():
    print(name, param.grad.shape)  # each param now has .grad filled in
```

### 2.3 The three-step training loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    # Step 1: zero gradients from previous batch
    optimizer.zero_grad()

    # Step 2: forward + backward
    out = model(batch["input"])
    loss = criterion(out, batch["target"])
    loss.backward()

    # Step 3: update parameters
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

**Why `zero_grad()` first?** PyTorch accumulates gradients — `.grad` is added to, not replaced, on each `.backward()`. Without zeroing, gradients from batch 1 contaminate batch 2.

### 2.4 `@torch.no_grad()` — turning off the graph

During evaluation and inference, you don't need gradients. Computing them wastes memory and time:

```python
# model.py:95
@torch.no_grad()
def synthesize(self, phoneme_ids, phoneme_lengths, n_steps=10):
    ...
```

`@torch.no_grad()` is a decorator that wraps the function body. Inside, no computation graph is built, no `.grad_fn` is stored on tensors, and memory usage is lower.

Alternative — use as a context manager:
```python
with torch.no_grad():
    mel = model.synthesize(...)
```

### 2.5 `detach()` — cutting a tensor from the graph

```python
# losses.py:12
loss = loss + F.l1_loss(f, r.detach())
```

`r.detach()` creates a new tensor with the same data but `requires_grad=False` and no connection to `r`'s computation graph. Used in feature matching loss so that gradients flow only through the fake (generator) path, not the real (discriminator) path — a GAN training standard.

### 2.6 `.item()` — extracting a scalar

```python
loss.item()   # float — extracts the Python float from a scalar tensor
```

Use `.item()` whenever you need the loss value for logging, comparison, or printing. Using the tensor directly keeps it in the graph and leaks memory.

```python
# Bad — keeps the graph alive
running_loss += loss

# Good — just the number
running_loss += loss.item()
```

### 2.7 Verifying a model trains

The first test after writing a model is confirming the loss has a gradient:

```python
# test_model_forward.py:12
assert "loss" in out
assert not torch.isnan(out["loss"])
assert out["loss"].requires_grad      # if this is False, the model is broken
```

If `requires_grad` is False, the model's forward pass broke the graph somewhere — usually by calling `.detach()` or `.item()` on an intermediate tensor by accident.

---

## Module 3 — `nn.Module`: The Building Block

### 3.1 The contract

Every neural network component in PyTorch must subclass `nn.Module`. The contract has two parts:

1. **`__init__`**: define all submodules and parameters
2. **`forward`**: define the computation

```python
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()           # REQUIRED — sets up internal PyTorch state
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))
```

**Why `super().__init__()` is required:** `nn.Module.__init__` sets up the internal dictionaries (`_parameters`, `_modules`, `_buffers`) that track everything. Without it, assigning `self.linear = nn.Linear(...)` silently stores a plain Python attribute instead of registering it as a submodule — meaning its parameters won't be found by `model.parameters()`, won't be moved by `.to(device)`, and won't be saved by checkpoint.

### 3.2 Calling a module — `__call__` vs `forward`

**Never call `forward` directly.** Always call the module like a function:

```python
layer = MyLayer(192, 192)
out = layer(x)        # correct — invokes __call__, which calls forward + hooks
out = layer.forward(x)  # wrong — bypasses hooks (including inference mode, etc.)
```

### 3.3 Parameters vs buffers

| Type | Created with | Included in `.parameters()` | Saved in state dict | Moved with `.to()` |
|---|---|---|---|---|
| Parameter | `nn.Parameter(...)` or assigned `nn.Module` | Yes | Yes | Yes |
| Buffer | `register_buffer(name, tensor)` | **No** | Yes | Yes |
| Plain tensor | `self.x = torch.zeros(...)` | No | **No** | **No** |

**In this project:** [encoder.py:17](../src/tts_pipeline/models/matcha/encoder.py#L17)
```python
self.register_buffer("pe", pe.unsqueeze(0))   # positional encoding — fixed, not learned
```

The sinusoidal positional encoding is precomputed once and stored as a buffer. It moves to GPU with the model, is saved in checkpoints, but has no gradient and doesn't appear in `optimizer.parameters()`.

### 3.4 `model.parameters()` and `named_parameters()`

```python
# Total parameter count:
total = sum(p.numel() for p in model.parameters())
print(f"{total:,} parameters")

# Inspect each:
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
```

### 3.5 `train()` and `eval()` mode

```python
model.train()   # activates Dropout and BatchNorm training behavior
model.eval()    # deactivates Dropout (pass-through), uses running stats in BatchNorm
```

This is not about gradients — it's about layer behavior. `@torch.no_grad()` controls gradients; `model.eval()` controls layer behavior. **Both are needed during inference:**

```python
model.eval()
with torch.no_grad():
    output = model(x)
```

**In this project:** [callbacks.py:11](../src/tts_pipeline/training/callbacks.py#L11)
```python
pl_module.eval()
with torch.no_grad():
    mel = pl_module.model.synthesize(...)   # inference during validation logging
pl_module.train()                           # restore training mode after
```

### 3.6 `model.apply(fn)` — weight initialization

```python
# hifigan.py:9-10
def _weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, 0.01)

# hifigan.py:35
self.apply(_weights_init)   # applies _weights_init to every submodule recursively
```

`model.apply(fn)` calls `fn(module)` on every submodule including `self`. Use it to set initial weights uniformly across a complex architecture.

---

## Module 4 — Core Layers

### 4.1 `nn.Embedding` — token lookup table

```python
# encoder.py:60
self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
```

An embedding is a learnable lookup table: each integer token ID maps to a vector of size `dim`.

```
Input:    phoneme_ids (B, T_text) — integers in [0, vocab_size)
Output:   (B, T_text, dim)       — each integer replaced by its embedding vector
```

`padding_idx=0`: the row for token 0 is forced to be all zeros and receives no gradient. This is the PAD token — it should not contribute to learning.

### 4.2 `nn.Linear` — fully connected layer

```python
# decoder.py:23-24
self.proj = nn.Sequential(
    nn.Linear(dim, dim * 4),   # weight: (dim*4, dim), bias: (dim*4,)
    nn.Mish(),
    nn.Linear(dim * 4, dim),
)
```

`nn.Linear(in, out)` computes `output = input @ weight.T + bias`.

```
Input:  (B, ..., in_features)
Output: (B, ..., out_features)
```

It operates on the **last dimension** only — all preceding dimensions are treated as batch dims. This is different from `nn.Conv1d` (see below).

**Pointwise convolution as projection:** [decoder.py:38-39](../src/tts_pipeline/models/matcha/decoder.py#L38-L39)
```python
self.cond_proj = nn.Conv1d(encoder_dim, channels, 1)   # kernel_size=1
```

A `Conv1d` with `kernel_size=1` is mathematically identical to `nn.Linear` applied at each time step. It's preferred here because the input is already in `(B, C, T)` format (channels-first) and avoids transposing.

### 4.3 `nn.Conv1d` — 1D convolution

```python
# encoder.py:27-29
self.net = nn.Sequential(
    nn.Conv1d(dim, dim * 4, kernel_size, padding=pad),   # expand
    nn.GELU(),
    nn.Conv1d(dim * 4, dim, kernel_size, padding=pad),   # contract
)
```

```
Input:   (B, C_in,  L)   — batch, channels, sequence length
Output:  (B, C_out, L)   — same L if padding = (kernel_size - 1) // 2
```

The **channel dimension is dimension 1** for `Conv1d`. This clashes with the usual `(B, T, D)` sequence format. The project handles this by transposing:

```python
# ConvFFN.forward, encoder.py:35
return self.norm(x + self.net(x.transpose(1, 2)).transpose(1, 2))
#                                 ↑                         ↑
#                          (B,T,D)→(B,D,T)            (B,D,T)→(B,T,D)
```

**Padding formula:**
```python
pad = (kernel_size - 1) // 2   # "same" padding — output length == input length
```

With `kernel_size=3, pad=1`: input length L → output length L. No shrinking.

**Dilation** in HiFi-GAN:
```python
# hifigan.py:13
nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=d*(kernel_size-1)//2)
```

Dilation `d` inserts `d-1` zeros between each kernel element, expanding the receptive field without increasing parameters:
```
dilation=1: [·  ·  ·]  covers 3 positions
dilation=3: [·     ·     ·]  covers 7 positions, same 3 parameters
```

### 4.4 `nn.ConvTranspose1d` — transposed convolution (upsampling)

```python
# hifigan.py:31
nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k-u)//2)
```

The inverse of `Conv1d`: expands the time dimension by `stride`:
```
Input:   (B, C_in,  L)
Output:  (B, C_out, L * stride)   approximately
```

HiFi-GAN uses four upsampling stages with strides `[8, 8, 2, 2]`:
```
mel:     (B, 80, T_mel)
after 1: (B, 256, T_mel*8)
after 2: (B, 128, T_mel*64)
after 3: (B, 64,  T_mel*128)
after 4: (B, 32,  T_mel*256)    ≈ T_audio (since hop_length=256)
```

Total upsample factor: 8×8×2×2 = 256 = `hop_length`. This is not a coincidence — HiFi-GAN is designed to invert the mel spectrogram's hop_length.

### 4.5 Normalization layers

**`nn.LayerNorm(dim)`** — normalizes across the feature dimension for each token:
```
Input:  (B, T, dim)
Output: (B, T, dim) — each (B, T) vector normalized to zero mean, unit variance
```
Most stable for Transformer-based models. No difference between train/eval.

**`nn.GroupNorm(num_groups, num_channels)`** — divides channels into groups, normalizes within each:
```python
# decoder.py:14
self.norm1 = nn.GroupNorm(8, channels)   # 8 groups of (channels/8) each
```
```
Input:  (B, C, L)   — channels-first format
Output: (B, C, L)
```
Used in the flow decoder because the input is in `(B, C, T)` format (channels first for Conv1d). `LayerNorm` expects `(B, T, D)` (channels last), so `GroupNorm` is the natural choice for Conv1d pipelines.

**`nn.utils.weight_norm(module)`** — wraps a Conv or Linear layer:
```python
# hifigan.py:13
nn.utils.weight_norm(nn.Conv1d(...))
```
Reparametrizes the weight matrix as `weight = g * (v / ||v||)`, where `g` is a scalar and `v` is the direction. This decouples magnitude from direction, improving gradient flow and training stability in GANs.

---

## Module 5 — Containers: `Sequential`, `ModuleList`, `ModuleDict`

### 5.1 `nn.Sequential`

For simple chains where the output of one layer is the input to the next:

```python
# decoder.py:22-25
self.proj = nn.Sequential(
    nn.Linear(dim, dim * 4),
    nn.Mish(),
    nn.Linear(dim * 4, dim),
)
```

`Sequential` is called like a single module: `self.proj(x)` passes `x` through all layers in order.

### 5.2 `nn.ModuleList` — dynamic lists of modules

When you need a list of modules but the list length varies, or you need to access them by index:

```python
# encoder.py:62-63
self.layers = nn.ModuleList(
    [TransformerEncoderLayer(dim, n_heads, kernel_size, dropout) for _ in range(n_layers)]
)

# encoder.py:83-84
for layer in self.layers:
    x = layer(x, key_padding_mask=mask)
```

**Why not a plain Python list?** `self.layers = [layer1, layer2]` stores modules in a list, but Python lists are not tracked by `nn.Module`. The layers' parameters would NOT appear in `model.parameters()`, would NOT be moved by `.to(device)`, and would NOT be saved in checkpoints.

**The rule:** any `nn.Module` you want to use must be assigned to a class attribute or registered explicitly. Use `nn.ModuleList` for lists, `nn.ModuleDict` for named dicts.

### 5.3 `nn.ModuleDict`

```python
phonemizers = nn.ModuleDict({
    "vi": ViAcousticModel(),
    "en": EnAcousticModel(),
})
output = phonemizers[language](x)   # dynamic dispatch
```

Not used in this project but useful for multi-language or multi-speaker variants.

---

## Module 6 — Sinusoidal Positional Encoding

### 6.1 The problem

Attention has no inherent sense of order — it treats the input as a set, not a sequence. Given the same tokens in different order, it produces the same output. Positional encoding injects position information.

### 6.2 The formula

```python
# encoder.py:10-20
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)                              # (max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()            # (max_len, 1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)  # even dims: sin
        pe[:, 1::2] = torch.cos(pos * div)  # odd  dims: cos
        self.register_buffer("pe", pe.unsqueeze(0))                 # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]  # add encoding for the first T positions
```

For each position `pos` and each dimension pair `(2i, 2i+1)`:
```
pe[pos, 2i]   = sin(pos / 10000^(2i/dim))
pe[pos, 2i+1] = cos(pos / 10000^(2i/dim))
```

Why this works:
- **Uniqueness:** each position gets a different pattern
- **Smoothness:** nearby positions have similar patterns
- **Extrapolation:** the formula works for positions beyond the training set

### 6.3 `register_buffer` — non-learnable, part of model state

```python
self.register_buffer("pe", pe.unsqueeze(0))
```

The positional encoding is precomputed and fixed — it's not learned. But it must:
- Move to GPU when `model.to("cuda")` is called
- Be saved and restored in checkpoints

`register_buffer` achieves exactly this. Access it as `self.pe` in `forward`. It appears in `model.state_dict()` but not in `model.parameters()`.

---

## Module 7 — Multi-Head Attention

### 7.1 Self-attention concept

Given a sequence of `T` vectors, attention asks: "for each position, which other positions are most relevant?"

It computes three projections of the input: **Query** (what I'm looking for), **Key** (what I have to offer), **Value** (what I return if selected). The output is a weighted sum of Values, where weights come from Query-Key similarities.

Multi-head attention runs this in parallel with `n_heads` different projection matrices, then concatenates and projects the results.

### 7.2 `nn.MultiheadAttention` in code

```python
# encoder.py:40
self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
```

```python
# encoder.py:44
attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
#                        ↑  ↑  ↑
#                     query key value  — all same for self-attention
```

`batch_first=True` means input shape is `(B, T, dim)` rather than `(T, B, dim)`. Always use `batch_first=True` in modern code.

Return value: `(attn_output, attn_weights)`. The weights are discarded here (the `_`).

### 7.3 Key padding mask

```python
# encoder.py:77
mask = torch.arange(T, device=phoneme_ids.device).unsqueeze(0) >= phoneme_lengths.unsqueeze(1)
# mask[b, t] = True if position t is padding for sequence b
```

Step by step for B=2, T=8, `phoneme_lengths=[6, 4]`:
```
torch.arange(8):            [0, 1, 2, 3, 4, 5, 6, 7]
.unsqueeze(0):              [[0, 1, 2, 3, 4, 5, 6, 7]]   shape (1, 8)
phoneme_lengths.unsqueeze(1): [[6], [4]]                  shape (2, 1)
>= comparison:
  row 0: [F, F, F, F, F, F, T, T]   positions 6,7 are padding
  row 1: [F, F, F, F, T, T, T, T]   positions 4-7 are padding
```

`True` positions are **ignored** by attention — they don't participate in the softmax.

### 7.4 The residual + LayerNorm pattern

```python
# encoder.py:44-45
attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
x = self.attn_norm(x + attn_out)   # residual connection, then normalize
```

The **residual connection** (`x + attn_out`) adds the input to the output before normalization. This:
1. Allows gradients to flow directly to early layers (solves vanishing gradient)
2. Gives the layer the option to make small refinements rather than full replacements
3. Enables training very deep networks (100+ layers in GPT-3)

This `residual → norm` pattern (post-norm) appears in every Transformer layer.

---

## Module 8 — Activations

### 8.1 Activations used in this project

| Activation | Location | Formula | When to use |
|---|---|---|---|
| `nn.GELU()` | Encoder ConvFFN | Smooth approximation to ReLU | Default in Transformers |
| `nn.Mish()` / `F.mish` | Flow decoder, time embedding | `x * tanh(softplus(x))` | Alternative to GELU, often better in small models |
| `F.leaky_relu(x, 0.1)` | HiFi-GAN | max(0.1x, x) | GANs — avoids dead neurons |
| `torch.tanh` | HiFi-GAN output | `tanh(x)` → [-1, 1] | Output squashing for audio |
| `nn.ReLU()` | Duration predictor | max(0, x) | Simple, fast |

### 8.2 `nn.X()` vs `F.X` — module vs functional

```python
# Module version — has state (can be placed in Sequential, moved to device)
self.act = nn.GELU()
out = self.act(x)

# Functional version — stateless (call directly in forward)
import torch.nn.functional as F
out = F.gelu(x)
```

For stateless activations (GELU, ReLU, Mish have no learnable parameters), both are identical in computation. Use `nn.X()` in `Sequential`, use `F.X` in `forward` when you prefer brevity.

### 8.3 Why `F.leaky_relu(x, 0.1)` in HiFi-GAN

Standard ReLU zeros out negative values, killing gradient flow for neurons that output negatives. A "dead" ReLU neuron never activates again once it's negative, permanently dropping from training. LeakyReLU with slope 0.1 keeps a small gradient for negatives — critical for GAN discriminators that need every neuron active.

---

## Module 9 — Tensor Operations for Sequence Modeling

### 9.1 The shape dimension conflict: Conv1d vs Attention

This is the most common source of confusion when reading TTS code. Two conventions coexist:

| Convention | Shape | Used by |
|---|---|---|
| Channels-last | `(B, T, D)` | `nn.MultiheadAttention`, `nn.Linear`, `nn.LayerNorm` |
| Channels-first | `(B, D, T)` | `nn.Conv1d`, `nn.ConvTranspose1d`, `nn.GroupNorm` |

When data flows between these, you transpose:

```python
# encoder.py:35 — ConvFFN forward
return self.norm(x + self.net(x.transpose(1, 2)).transpose(1, 2))
#                   x: (B, T, D)
#                       x.transpose(1, 2): (B, D, T) ← for Conv1d
#                       self.net(...):     (B, D, T)
#                       .transpose(1,2):  (B, T, D) ← back for LayerNorm
```

```python
# decoder.py:48
cond = self.cond_proj(condition.transpose(1, 2))   # (B,T,D) → (B,D,T) for Conv1d(1)
```

`transpose(1, 2)` swaps dimensions 1 and 2. It's not a copy — it returns a view of the same memory with different strides.

### 9.2 `masked_fill` — zeroing out padding

```python
# duration_predictor.py:22-23
log_dur = self.proj(h.transpose(1, 2)).squeeze(-1)  # (B, T_text)
if mask is not None:
    log_dur = log_dur.masked_fill(mask, 0.0)

# model.py:87
dur_loss = ((log_durations - log_gt) ** 2).masked_fill(enc_mask, 0.0).mean()
```

`tensor.masked_fill(mask, value)` replaces positions where `mask == True` with `value`. This zeros out loss contributions from padding tokens — you don't want the model to "learn" to predict durations for PAD tokens.

**Shape requirement:** `mask` must be broadcastable to `tensor.shape`.

### 9.3 Time conditioning by broadcasting

```python
# decoder.py:47-49
time_emb = self.time_emb(t)           # (B, channels)
cond = self.cond_proj(condition.T)    # (B, channels, T_mel)
h = self.input_proj(x_t) + cond
h = h + time_emb[:, :, None]         # (B, channels, 1) broadcasts over T_mel
```

`time_emb[:, :, None]` adds a dimension at position 2 turning `(B, channels)` into `(B, channels, 1)`. This broadcasts to match `h.shape = (B, channels, T_mel)`, adding the same time embedding to every time step.

### 9.4 Skip connections in the U-Net decoder

```python
# decoder.py:50-53
skips: list[torch.Tensor] = []
for layer in self.encoder_layers:
    h = layer(h)
    skips.append(h)       # save intermediate activations

h = self.middle(h)

for layer, skip in zip(self.decoder_layers, reversed(skips)):
    h = layer(h + skip)   # add saved activation before passing to next layer
```

This is the U-Net skip connection pattern. The encoder path saves progressively deeper features; the decoder path combines them with the deepest features, restoring spatial detail. For audio, it helps the decoder remember fine-grained spectral structure from early layers.

---

## Module 10 — Optimizers and Learning Rate Schedulers

### 10.1 `AdamW` — the standard for Transformers

```python
# model.py:150
opt = AdamW(self.parameters(), lr=self._training_cfg["learning_rate"], weight_decay=1e-2)
```

`AdamW` is Adam with **decoupled weight decay**. Standard Adam applies weight decay through the gradient, which interacts badly with Adam's adaptive learning rate. `AdamW` applies it directly to the weights:

```
weight_update = -lr * (adam_step + weight_decay * weight)
```

| Parameter | Default | Role |
|---|---|---|
| `lr` | 1e-4 | Step size |
| `weight_decay` | 1e-2 | L2 regularization strength |
| `betas` | (0.9, 0.999) | Momentum for first/second moment |
| `eps` | 1e-8 | Numerical stability |

**In this project:** [params.yaml:42](../params.yaml#L42): `learning_rate: 1.0e-4`

### 10.2 Learning rate scheduling — warmup + cosine decay

Training with a high learning rate from step 0 causes instability — gradients are large when parameters are random, and a big update destroys structure before it forms. **Warmup** gradually increases the LR to the target value over the first N steps.

```python
# model.py:151-153
warmup = LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
cosine = CosineAnnealingLR(opt, T_max=max_epochs)
scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
```

```
LR
 ▲
 │          ╭──────────────────────╮
 │        ╭╯                        ╰──────────╮
 │      ╭╯                                       ╰─────
 │    ╭╯
 │───╯  (warmup)            (cosine decay)
 ╰──────────────────────────────────────────────────▶ step
      0    warmup_steps                         max_steps
```

`SequentialLR` runs `warmup` for the first `warmup_steps` steps, then switches to `cosine`.

**In this project:** [params.yaml:43-44](../params.yaml#L43-L44): `warmup_steps: 1000`, `max_epochs: 1000`

### 10.3 Gradient clipping

```python
# trainer.py:29
trainer = L.Trainer(gradient_clip_val=cfg.training.grad_clip_val, ...)
```

Gradient clipping caps the global gradient norm before the optimizer step:

```python
# What Lightning does internally (equivalent):
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Without clipping, occasional large gradients (from numerical instability) can cause a single catastrophic update that destroys hours of training. With `grad_clip_val=1.0`, any gradient update larger than 1.0 in L2 norm is scaled down.

**In this project:** [params.yaml:44](../params.yaml#L44): `grad_clip_val: 1.0`

### 10.4 Gradient accumulation

```python
# trainer.py:30
accumulate_grad_batches=cfg.training.accumulate_grad_batches
```

When a single batch doesn't fit in GPU memory, you can process N smaller batches without calling `optimizer.step()`, effectively training with batch size `N * mini_batch_size`:

```python
# What Lightning does internally:
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulate_grad_batches
    loss.backward()
    if (i + 1) % accumulate_grad_batches == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**In this project:** [params.yaml:45](../params.yaml#L45): `accumulate_grad_batches: 1` (no accumulation by default — a single GPU with enough memory)

---

## Module 11 — Mixed Precision Training

### 11.1 Floating-point formats

| Format | Bits | Exponent bits | Mantissa bits | Dynamic range |
|---|---|---|---|---|
| float32 (FP32) | 32 | 8 | 23 | ~1e-38 to 1e38 |
| float16 (FP16) | 16 | 5 | 10 | ~6e-8 to 65504 |
| bfloat16 (BF16) | 16 | 8 | 7 | ~1e-38 to 1e38 |

**BF16 vs FP16:** BF16 has the same exponent range as FP32 (safe for gradients that can be very large or small). FP16 has overflow problems with gradients beyond 65504 and requires a loss scaling trick to compensate. BF16 needs no loss scaling.

**A100 / H100 GPUs have native BF16 Tensor Cores.** BF16 matrix multiplications run at the same hardware speed as FP16 but with FP32-level numerical stability.

### 11.2 Mixed precision: the idea

Keep model weights in FP32 (high precision) but compute forward and backward passes in BF16 (fast). At each step:

```
Forward:  BF16 computation (fast)
Backward: BF16 gradients computed
Update:   FP32 master weights updated with BF16 gradients
```

### 11.3 `precision="bf16-mixed"` in Lightning

```python
# trainer.py:30
trainer = L.Trainer(precision=cfg.training.precision, ...)
# params.yaml: precision: "bf16-mixed"
```

Lightning handles everything automatically. You don't write `autocast` or `GradScaler` manually. The full manual equivalent would be:

```python
# What happens under the hood (simplified):
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(batch)
    loss = criterion(output, target)

loss.backward()
optimizer.step()
```

**In this project:** [params.yaml:52](../params.yaml#L52) (from earlier context): `precision: "bf16-mixed"`

### 11.4 When NOT to use mixed precision

- **CPU training:** no hardware acceleration for BF16 on most CPUs — FP32 is faster
- **Small models:** the overhead of casting tensors may exceed the compute savings
- **Operations with numerical sensitivity:** some loss functions (PESQ, certain normalizations) may need FP32 sub-expressions

---

## Module 12 — `torch.compile`

### 12.1 What it does

`torch.compile` (introduced in PyTorch 2.0) traces your model's computation graph and compiles it to optimized native kernels via the **Inductor** backend (which generates Triton or C++ code depending on the target):

```python
# trainer.py:23-24
if cfg.training.use_compile:
    pl_model = torch.compile(pl_model)
```

**Before compile:** PyTorch executes operations one at a time through Python — Python overhead, no kernel fusion.
**After compile:** the entire forward pass is one compiled artifact — kernel fusion, memory layout optimization, graph-level optimization.

Typical speedup: 10–30% on A100. Larger speedup on transformer-heavy models.

### 12.2 What it requires

- **PyTorch 2.0+** (this project uses 2.3.0+)
- A **GPU** (A100/H100 benefit most, but any CUDA GPU works)
- **Static shapes preferred**: dynamic shapes work but break the compiled graph, forcing recompilation
- **Python control flow limitations**: if/else that depends on tensor values during the forward pass can break tracing

### 12.3 First-batch overhead

The first forward pass after `torch.compile` triggers compilation — this can take 30–120 seconds. All subsequent passes use the compiled version. In a 1000-epoch training run, the compilation overhead is negligible.

### 12.4 `use_compile: false` is the safe default

```yaml
# params.yaml (from training section):
# use_compile is false by default — enable only on A100/H100 for production runs
```

Keep `use_compile=False` during development and debugging. Compiled models produce opaque error messages when something goes wrong.

---

## Module 13 — The Full Training Flow, End to End

Now trace a complete forward + backward pass through the project:

```
Input batch:
  phoneme_ids:    (B=16, T_text=50)     int64
  phoneme_lengths:(B=16,)               int64
  mel_targets:    (B=16, 80, T_mel=200) float32
  mel_lengths:    (B=16,)               int64

─── MatchaTTSLightning.training_step ────────────────────────────────────

1. model.encoder(phoneme_ids, phoneme_lengths):
   a. embedding(phoneme_ids)            → (16, 50, 192)   float32
   b. pos_enc(x)                        → (16, 50, 192)   adds sinusoidal
   c. 6× TransformerEncoderLayer:
      - MultiheadAttention + residual   → (16, 50, 192)
      - ConvFFN + residual              → (16, 50, 192)
   d. LayerNorm                         → (16, 50, 192)
   e. return enc_out, mask              → (16,50,192), (16,50) bool

2. duration_predictor(enc_out, mask):
   a. 2× [Conv1d → ReLU → LayerNorm]   → (16, 50, 256)
   b. Linear → squeeze                 → (16, 50)   log-durations
   c. masked_fill padding positions    → (16, 50)

3. Compute ground-truth durations from mel/text ratio
   gt_durations                        → (16, 50)   int

4. DurationPredictor.regulate(enc_out, gt_durations):
   Repeat each phoneme vector d times  → (16, T_mel, 192)

5. Flow matching sampling:
   t = torch.rand(16)                  → (16,)   uniform [0,1]
   x0 = torch.randn_like(mel_targets)  → (16, 80, 200)
   x_t, u_t = cfm.sample_xt(x0, mel_targets, t)  → both (16, 80, 200)

6. flow_decoder(x_t, t, regulated):
   a. time embedding(t)                → (16, 256)
   b. cond_proj(regulated.T)           → (16, 256, 200)
   c. input_proj(x_t) + cond + t_emb  → (16, 256, 200)
   d. U-Net encoder/decoder residuals  → (16, 256, 200)
   e. out_proj                         → (16, 80, 200)   predicted vector field

7. Losses:
   flow_loss = MSE(pred_vf, u_t)       → scalar
   dur_loss  = masked MSE(log_dur, log_gt) → scalar
   loss = flow_loss + dur_loss          → scalar

─── Lightning orchestration ──────────────────────────────────────────────

8. optimizer.zero_grad()
9. loss.backward()                     ← gradients flow through entire graph
10. clip_grad_norm_(params, 1.0)
11. optimizer.step()                   ← AdamW updates all parameters
12. scheduler.step()                   ← LR warmup/decay
13. log_dict({"train/loss": loss, ...}) → MLflow
```

---

## Exercises

**Exercise 1** — Parameter count. Write a function `count_parameters(model: nn.Module) -> dict[str, int]` that returns the number of parameters in each top-level submodule of `MatchaTTS`. Run it and compare encoder vs decoder vs flow_decoder.

**Exercise 2** — Shape tracing. Write a standalone script that creates a `PhonemeEncoder(vocab_size=50, dim=32, n_heads=2, n_layers=2)` and calls `forward` with a batch of 3 sequences of length 10. Print `x.shape` at each step inside the encoder (embedding, pos_enc, after each layer, after norm).

**Exercise 3** — `register_buffer` behavior. Create a simple `nn.Module` that has both a learnable parameter and a buffer. Show that:
- Both appear in `model.state_dict()`
- Only the parameter appears in `list(model.parameters())`
- Both move when `model.to("cuda")` is called (if you have a GPU) or `model.to("meta")`

**Exercise 4** — Masked loss. Create a batch of 3 sequences of different lengths. Compute MSE loss with and without the padding mask and show that the masked version gives a lower (more correct) loss value.

**Exercise 5** — The `.transpose(1, 2)` pattern. Write a `ConvBlock` that takes `(B, T, D)` input, applies a `nn.Conv1d` internally, and returns `(B, T, D)`. Test it produces the same output shape for any input T.

**Exercise 6** — Verify mixed precision manually. Without Lightning, wrap a single forward pass in `torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)` and confirm that the weights remain float32 but intermediate activations become bfloat16.

---

## Mini Practice Project: `miniflow` — Tiny Flow-Matching Trainer

This is an independent project that builds a stripped-down version of the Matcha-TTS training loop from scratch. No TTS data. No audio. Just the neural architecture concepts and training loop.

**What it builds:** A tiny model that learns to map Gaussian noise to a target 2D distribution using conditional flow matching — the same mathematical idea as Matcha-TTS, but on toy data you can visualize.

**What it practices:**

| Course concept | Where in `miniflow` |
|---|---|
| `nn.Module` + `super().__init__()` | Every model class |
| `nn.Linear`, `nn.LayerNorm` | Encoder and decoder |
| `nn.ModuleList` | Stacked encoder layers |
| `register_buffer` | Sinusoidal time embedding freqs |
| Autograd: `.backward()`, `zero_grad()` | Training loop |
| `@torch.no_grad()` | Inference |
| `AdamW` + `LinearLR` + `CosineAnnealingLR` | Optimizer setup |
| Gradient clipping | `clip_grad_norm_` |
| Broadcasting `t[:, None]` | Time conditioning |
| `masked_fill` | Optional padding |
| Mixed precision | `torch.amp.autocast` |

**Time estimate:** 4–6 hours.

---

### Step 0 — What the model learns

Instead of mel spectrograms, the target distribution is samples from a mixture of 2D Gaussians (a "Swiss roll" pattern or a checkerboard). Noise `x_0 ~ N(0, I)` is mapped to target `x_1 ~ p_target` via learned Euler integration.

This is the exact same problem as Matcha-TTS, just in 2D instead of 80-dimensional mel space. You can visualize the learned distribution in a scatter plot.

---

### Step 1 — `model.py`

```python
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encode scalar time t ∈ [0,1] as a d-dimensional vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)    # not learned, moves with model
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → (B, dim)
        args = t[:, None] * self.freqs[None]   # (B, half) — broadcasting
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.proj(emb)


class ResidualBlock(nn.Module):
    """Single residual block: Linear → norm → activation → Linear + skip."""

    def __init__(self, dim: int, time_dim: int) -> None:
        super().__init__()
        self.time_proj = nn.Linear(time_dim, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.Mish(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, dim), t_emb: (B, time_dim)
        return x + self.net(x + self.time_proj(t_emb))


class VectorFieldNet(nn.Module):
    """
    Predicts the vector field v(x_t, t) for flow matching.
    Input:  noisy sample x_t (B, data_dim) + time t (B,)
    Output: predicted vector field (B, data_dim)
    """

    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, n_layers: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.layers = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, data_dim)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_t)          # (B, hidden_dim)
        t_emb = self.time_emb(t)          # (B, hidden_dim)
        for layer in self.layers:
            h = layer(h, t_emb)
        return self.output_proj(h)         # (B, data_dim)

    @torch.no_grad()
    def sample(self, n: int, n_steps: int = 100, device: str = "cpu") -> torch.Tensor:
        """Euler integration from noise to data. Returns (n, data_dim) samples."""
        x = torch.randn(n, self.output_proj.out_features, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((n,), i * dt, device=device)
            vf = self(x, t)
            x = x + dt * vf
        return x
```

---

### Step 2 — `data.py` — toy target distributions

```python
from __future__ import annotations

import torch
import math


def sample_checkerboard(n: int) -> torch.Tensor:
    """Sample from a checkerboard distribution in [-2, 2]^2."""
    x1 = torch.rand(n) * 4 - 2
    x2 = torch.rand(n) * 4 - 2
    mask = ((x1.floor() + x2.floor()) % 2 == 0)
    x1 = x1[mask][:n // 2]
    x2 = x2[mask][:n // 2]
    # Pad if needed
    while len(x1) < n:
        extra = sample_checkerboard(n - len(x1))
        x1 = torch.cat([x1, extra[:, 0]])
        x2 = torch.cat([x2, extra[:, 1]])
    return torch.stack([x1[:n], x2[:n]], dim=-1)


def sample_swiss_roll(n: int) -> torch.Tensor:
    """Sample from a 2D Swiss roll."""
    t = (3 * math.pi / 2) * (1 + 2 * torch.rand(n))
    x = t * torch.cos(t) / 10
    y = t * torch.sin(t) / 10
    return torch.stack([x, y], dim=-1)


def get_batch(n: int, distribution: str = "checkerboard") -> torch.Tensor:
    """Return n samples from the target distribution."""
    if distribution == "checkerboard":
        return sample_checkerboard(n)
    elif distribution == "swiss_roll":
        return sample_swiss_roll(n)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
```

---

### Step 3 — `train.py` — the training loop

```python
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from miniflow.data import get_batch
from miniflow.model import VectorFieldNet


def flow_matching_loss(
    model: VectorFieldNet,
    x1: torch.Tensor,
    sigma_min: float = 1e-4,
) -> torch.Tensor:
    """
    Sample x_t along the flow path and compute MSE on the vector field.
    This is the exact same loss as in model.py:83.
    """
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device)                    # uniform time
    x0 = torch.randn_like(x1)                              # noise

    # Interpolate: x_t = t*x1 + (1-t)*x0 + noise
    t_b = t[:, None]                                        # (B, 1) — broadcasts over dim
    mu_t = t_b * x1 + (1 - t_b) * x0
    sigma_t = sigma_min + (1 - sigma_min) * t_b
    x_t = mu_t + sigma_t * torch.randn_like(x1)

    # Target vector field (constant in OT-CFM):
    u_t = x1 - (1 - sigma_min) * x0

    pred_vf = model(x_t, t)
    return nn.functional.mse_loss(pred_vf, u_t)


def train(
    n_epochs: int = 500,
    batch_size: int = 512,
    lr: float = 3e-4,
    warmup_steps: int = 200,
    hidden_dim: int = 128,
    n_layers: int = 4,
    distribution: str = "checkerboard",
    use_amp: bool = False,
) -> VectorFieldNet:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    model = VectorFieldNet(data_dim=2, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup = LinearLR(opt, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=n_epochs)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scaler = torch.amp.GradScaler() if use_amp and device == "cuda" else None

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()

        x1 = get_batch(batch_size, distribution).to(device)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                loss = flow_matching_loss(model, x1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss = flow_matching_loss(model, x1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        scheduler.step()

        if epoch % 50 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d} | loss={loss.item():.4f} | lr={current_lr:.2e}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--distribution", default="checkerboard",
                        choices=["checkerboard", "swiss_roll"])
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    model = train(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        distribution=args.distribution,
        use_amp=args.amp,
    )
    torch.save(model.state_dict(), "miniflow_model.pt")
    print("Saved miniflow_model.pt")


if __name__ == "__main__":
    main()
```

---

### Step 4 — `eval.py` — visualize what the model learned

```python
from __future__ import annotations

import torch
import sys

from miniflow.model import VectorFieldNet
from miniflow.data import get_batch


def evaluate(model_path: str, distribution: str = "checkerboard") -> None:
    model = VectorFieldNet(data_dim=2, hidden_dim=128, n_layers=4)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Generate samples
    samples = model.sample(n=2000, n_steps=100, device="cpu").numpy()
    real = get_batch(2000, distribution).numpy()

    # ASCII scatter plot
    print("\nGenerated samples (should look like real distribution):")
    _ascii_scatter(samples, title="Generated")
    print("\nReal samples:")
    _ascii_scatter(real, title="Real")


def _ascii_scatter(points: "np.ndarray", title: str, width: int = 60, height: int = 20) -> None:
    import numpy as np

    x, y = points[:, 0], points[:, 1]
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5

    grid = [[" "] * width for _ in range(height)]
    for xi, yi in zip(x, y):
        col = int((xi - x_min) / (x_max - x_min) * (width - 1))
        row = int((1 - (yi - y_min) / (y_max - y_min)) * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = "·"

    print(f"{title}:")
    print("┌" + "─" * width + "┐")
    for row in grid:
        print("│" + "".join(row) + "│")
    print("└" + "─" * width + "┘")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "miniflow_model.pt"
    distribution = sys.argv[2] if len(sys.argv) > 2 else "checkerboard"
    evaluate(path, distribution)
```

---

### Step 5 — Run it

```bash
pip install torch

python -m miniflow.train --epochs 500 --distribution checkerboard
python -m miniflow.eval miniflow_model.pt checkerboard
```

Expected behavior:
- Loss decreases from ~1.0 to ~0.05 over 500 epochs
- The ASCII scatter of generated samples should resemble a checkerboard pattern
- Swiss roll should form a spiral shape

---

### Step 6 — Experiments to run

Each of these changes one concept at a time, which is how you build intuition:

| Experiment | What to change | What to observe |
|---|---|---|
| Remove `zero_grad()` | Comment out `opt.zero_grad()` | Loss diverges immediately |
| Remove gradient clip | Comment out `clip_grad_norm_` | Loss becomes unstable (NaN possible) |
| Remove warmup | Replace with fixed LR | Slower early convergence, sometimes diverges |
| Replace `AdamW` with SGD | `torch.optim.SGD(lr=0.01)` | Much harder to converge on this problem |
| Use a plain Python list instead of `ModuleList` | `self.layers = [ResidualBlock(...)]` | Parameters not updated, loss doesn't decrease |
| Set `requires_grad=False` on one layer | `model.input_proj.weight.requires_grad = False` | That layer stops learning — check with `param.grad` |
| Remove the time embedding | Replace `t_emb` with zeros | Model can't distinguish t=0 from t=1, fails |

---

### Completion Checklist

- [ ] Explain why `super().__init__()` is required in every `nn.Module` subclass
- [ ] List three consequences of using a plain Python `list` instead of `nn.ModuleList`
- [ ] Write the three-step training loop (`zero_grad` → `backward` → `step`) from memory
- [ ] Explain the difference between `model.eval()` and `torch.no_grad()` — when do you need both?
- [ ] Explain what `masked_fill(mask, 0.0)` does in the duration loss and why it's needed
- [ ] Trace through `VectorFieldNet.forward` and write the shape at every intermediate step
- [ ] Run the "remove `nn.ModuleList`" experiment and confirm the loss doesn't decrease
- [ ] Explain in one sentence why BF16 is preferred over FP16 for training on A100/H100
- [ ] Run the full `miniflow` pipeline and produce an ASCII scatter plot that looks like the target distribution