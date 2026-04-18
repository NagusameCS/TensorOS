# ==============================================================================
# TensorOS — Pseudocode Language Guide
# ==============================================================================

## Overview

Pseudocode is TensorOS's default programming language. The goal is readable model and tensor code that still lowers cleanly to the runtime's tensor operations.

Inspired by [NaguSamecs' Pseudocode](https://github.com/NaguSamecs/Pseudocode).

## Syntax

### Variables and Types

```pseudocode
x = 42
name = "TensorOS"
pi = 3.14159
active = true
```

Types are inferred. Tensors are first-class:

```pseudocode
weights = tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
biases = tensor([0.1, 0.2])
```

### Tensor Operations

```pseudocode
C = matmul(A, B)
D = A + B
E = relu(D)
F = softmax(logits)
G = layernorm(x, gamma, beta)
```

### Model Definition

```pseudocode
model MyTransformer:
    layer self_attention(Q, K, V):
        scores = matmul(Q, transpose(K)) / sqrt(d_k)
        attn = softmax(scores)
        return matmul(attn, V)

    layer feed_forward(x):
        h = gelu(matmul(x, W1) + b1)
        return matmul(h, W2) + b2

    layer transformer_block(x):
        a = self_attention(x, x, x)
        x = layernorm(x + a)
        f = feed_forward(x)
        return layernorm(x + f)
```

### Loading Models

```pseudocode
load "llama-3-8b" as llm
load "stable-diffusion-xl" as sdxl
load "./my-model.safetensors" as custom
```

### Inference

```pseudocode
result = infer llm with "What is the meaning of life?"
print result

image = infer sdxl with "a cat on the moon, oil painting"
save image to "output.png"
```

### Training

```pseudocode
train llm on "dataset.jsonl":
    epochs = 3
    learning_rate = 0.0003
    batch_size = 32
    optimizer = adamw
    save every 500 steps
    eval every 100 steps
```

### Deployment

```pseudocode
deploy llm on port 8080
deploy sdxl on port 8081
```

### Git Operations

```pseudocode
git init
git commit "initial model checkpoint"
git branch "experiment-1"
git push
```

### Pipelines

```pseudocode
pipeline translate_and_summarize:
    step 1: infer translator with input
    step 2: infer summarizer with step1.output
    return step2.output
```

### Control Flow

```pseudocode
if accuracy > 0.95:
    print "Model converged!"
    deploy model on port 8080
else:
    train model on "more-data.jsonl"

for i in range(10):
    result = infer model with prompts[i]
    print result

while loss > 0.01:
    train model for 100 steps
```

### Functions

```pseudocode
function preprocess(text):
    tokens = tokenize(text)
    embeddings = embed(tokens)
    return embeddings

function evaluate(model, dataset):
    total = 0
    correct = 0
    for sample in dataset:
        pred = infer model with sample.input
        if pred == sample.label:
            correct = correct + 1
        total = total + 1
    return correct / total
```

## AI-Specific Keywords

| Keyword | Purpose |
|---------|---------|
| `model` | Define a model architecture |
| `layer` | Define a layer within a model |
| `tensor` | Create a tensor literal |
| `train` | Start training |
| `infer` | Run inference |
| `load` | Load a model from file or registry |
| `save` | Save model/tensor to file |
| `deploy` | Deploy model as a service |
| `pipeline` | Define a multi-model pipeline |
| `monitor` | Access system monitoring |
| `git` | Git operations |

## Supported Tensor Operations

| Operation | Syntax |
|-----------|--------|
| Matrix multiply | `matmul(A, B)` |
| Element-wise add | `A + B` |
| Element-wise multiply | `A * B` |
| ReLU | `relu(x)` |
| GELU | `gelu(x)` |
| SiLU | `silu(x)` |
| Softmax | `softmax(x)` |
| Layer norm | `layernorm(x, gamma, beta)` |
| Transpose | `transpose(x)` |
| Reshape | `reshape(x, shape)` |
| Concatenate | `concat(a, b, axis)` |
| Convolution | `conv2d(input, kernel)` |
| Embedding | `embed(indices)` |

## JIT Compilation Tiers

Pseudocode uses a 4-tier JIT compilation strategy:

1. **Tier 0 — Interpreter**: Tree-walking execution (first run)
2. **Tier 1 — Basic JIT**: Direct TIR → machine code (after 10 calls)
3. **Tier 2 — Optimized**: Op fusion, precision auto-downgrade (after 100 calls)
4. **Tier 3 — Fully Optimized**: Full optimization pipeline (after 1000 calls)

Hot code paths automatically promote to higher tiers.

## Backend Selection

The runtime automatically selects CPU, GPU, or TPU based on:

- **Tensor size**: Small tensors (< 4KB) → CPU; large → GPU
- **Operation type**: matmul, attention, conv2d → GPU; small elementwise → CPU
- **Device load**: If GPU is busy, may fall back to CPU
- **Available hardware**: Auto-detects NVIDIA/AMD/Intel GPUs

No manual device placement needed — but you can override:

```pseudocode
with device "gpu:0":
    result = matmul(A, B)

with device "cpu":
    small_result = relu(x)
```
