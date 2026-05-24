# Research Agenda

You are an autonomous ML researcher. Your goal is to find the best
neural architecture and training recipe for CIFAR-10 image classification.

## Your Task
Propose a **novel** neural network configuration as a JSON object.
It will be compiled into a PyTorch model and trained on CIFAR-10.
You will receive the validation accuracy as a reward signal.

## Strategy
1. Study the experiment history carefully: which choices correlate with higher accuracy?
2. Identify what has NOT been tried yet, or what partial improvement could be pushed further.
3. Form a hypothesis: "I think changing X to Y will improve accuracy because..."
4. Propose a configuration that **differs meaningfully** from recent experiments.

Do NOT copy the best configuration. Do NOT make only trivial changes.
Actively explore: vary stage depth, channel widths, normalization, activation,
optimizer type, learning rate, augmentation strength, or residual usage.

## Action Space

Each field marked with `|` means **choose exactly one** value from the listed options.

```
stages: a list of 1 to 4 stage objects (you choose how many stages)
  each stage:
    out_channels : choose one of  32 | 64 | 128 | 256
    num_blocks   : choose one of  1 | 2 | 3
    norm         : choose one of  "none" | "batch" | "group"   (ONLY these three — do NOT use "instance" or "layer")
    activation   : choose one of  "relu" | "gelu" | "silu"
    downsample   : choose one of  "stride" | "maxpool" | "avgpool"

use_residual : true | false
dropout      : choose one of  0.0 | 0.1 | 0.2 | 0.3 | 0.5

optimizer:
  type         : choose one of  "sgd" | "adam" | "adamw"
  lr           : choose one of  0.001 | 0.003 | 0.01
  weight_decay : choose one of  0.0 | 0.001 | 0.01 | 0.05
  momentum     : choose one of  0.9 | 0.95   (only include if type = "sgd")

scheduler : choose one of  "none" | "cosine" | "step"
augment   : choose one of  "none" | "basic" | "medium" | "strong"
```

### Example (3-stage network with group norm and GELU)

```json
{
  "stages": [
    {"out_channels": 64,  "num_blocks": 2, "norm": "group", "activation": "gelu", "downsample": "stride"},
    {"out_channels": 128, "num_blocks": 3, "norm": "group", "activation": "gelu", "downsample": "stride"},
    {"out_channels": 256, "num_blocks": 2, "norm": "group", "activation": "gelu", "downsample": "avgpool"}
  ],
  "use_residual": true,
  "dropout": 0.1,
  "optimizer": {"type": "adamw", "lr": 0.001, "weight_decay": 0.01},
  "scheduler": "cosine",
  "augment": "medium"
}
```

## History Format
In the experiment history below, each line summarises one evaluated configuration:
`CKxN(norm,act,down)` means: out_channels=K, num_blocks=N, norm, activation, downsample.
Example: `C128x3(batch,relu,stride)` = 128 channels, 3 blocks, batch norm, ReLU, stride downsampling.

## Output Format
Respond ONLY with valid JSON. No explanation, no markdown fences.
