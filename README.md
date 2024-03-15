# 🎨 Text-to-Image Generator (Replicate API)

Generate images from text prompts using **Replicate** — a developer-friendly SaaS with free tier. Uses **Stable Diffusion XL** via simple API calls.

> No GPU required. Just set your API token and go.

## 🔒 Why This Approach?

- ✅ **Free Tier**: Get free credits on signup (~200-500 images)
- ✅ **No GPU Needed**: Runs in Replicate's cloud
- ✅ **Simple Setup**: Just an API token, no complex provisioning
- ✅ **Multiple Models**: SDXL, SD 1.5, and more
- ✅ **Pay-per-use**: No monthly fees after free credits

## 🚀 Quick Start

### 1. Get API Token

1. Sign up at https://replicate.com
2. Go to Account → API Tokens
3. Copy your token

### 2. Install & Run

```bash
pip install -r requirements.txt

export REPLICATE_API_TOKEN="your_token_here"

python generate_replicate.py "a beautiful sunset over mountains, oil painting style" -o sunset.png
```

### Advanced Usage

```bash
# High quality SDXL
python generate_replicate.py "a cyberpunk city with neon lights" \
    --model sdxl \
    --steps 40 \
    --cfg-scale 8.0 \
    --width 1024 \
    --height 1024

# Faster with SD 1.5
python generate_replicate.py "a cute cat" --model sd15 --steps 25

# Multiple outputs
python generate_replicate.py "abstract art" --num-outputs 4 --seed 42

# List available models
python generate_replicate.py --list-models
```

### Web UI

```bash
python gradio_ui.py
# Open http://localhost:7860
```

### Batch Generation

```bash
python batch_generate.py
```

## 📦 Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.8+ and an internet connection. No GPU needed.

## 🎯 Usage Examples

### Python Code

```python
from generate_replicate import ReplicateImageGenerator

generator = ReplicateImageGenerator()

images = generator.generate_image(
    prompt="a serene lake at dawn, watercolor style",
    model="sdxl",
    width=1024,
    height=1024,
    num_inference_steps=30
)

images[0].save("lake.png")
```

## 🎨 Tips for Better Results

### Prompt Engineering

**Good prompts include:**
- Subject: "a red dragon"
- Style: "oil painting", "photorealistic", "anime style"
- Details: "detailed", "8k", "highly detailed"
- Lighting: "dramatic lighting", "soft light", "golden hour"

**Examples:**
```
✓ "a majestic red dragon perched on a mountain peak, fantasy art, detailed scales, dramatic lighting, 4k"
✗ "dragon"
```

### Negative Prompts

```bash
python generate_replicate.py "portrait of a warrior" \
    --negative "blurry, bad quality, distorted, watermark"
```

### Parameters Guide

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| Steps | 1-500 | 30 | More = better quality, slower |
| CFG Scale | 1-20 | 7.5 | Higher = stricter prompt adherence |
| Width/Height | 128-1536 | 1024 | Image dimensions |

## 🔧 Available Models

| Model | Resolution | Best For |
|-------|-----------|----------|
| **sdxl** (default) | 1024×1024 | High quality, general purpose |
| **sd15** | 512×512 | Fast generation, lower cost |

## 💰 Pricing

- **Free credits** on signup (enough for ~200-500 images)
- After that: ~$0.003-0.01 per image depending on model

## 🐛 Troubleshooting

### "API token required"
```bash
export REPLICATE_API_TOKEN="r8_your_token_here"
```

### "Model not found"
Use `--model sdxl` or `--model sd15`

### Rate limiting
Free tier has rate limits. Wait a moment and retry.
