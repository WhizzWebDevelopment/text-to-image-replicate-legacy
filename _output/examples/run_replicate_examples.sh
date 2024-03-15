#!/bin/bash
# Replicate API Examples
# Replicate text-to-image

set -e

OUTPUT_DIR="_output/replicate"
echo "🟢 Replicate API Examples (Small SaaS with Free Tier)"
echo "======================================================"
echo ""

# Check for API token
if [ -z "$REPLICATE_API_TOKEN" ]; then
    echo "❌ Error: REPLICATE_API_TOKEN not set"
    echo "Get your token at: https://replicate.com/account/api-tokens"
    echo "Then run: export REPLICATE_API_TOKEN='r8_your_token_here'"
    exit 1
fi

echo "✓ API token found"
echo ""

# Example 1: Basic SDXL generation
echo "Example 1: Basic SDXL generation (1024×1024)"
python generate_replicate.py \
    "a serene mountain landscape at sunset, professional photography, 8k" \
    --model sdxl \
    --output "$OUTPUT_DIR/example1_landscape.png"

# Example 2: Portrait with negative prompt
echo ""
echo "Example 2: Portrait with negative prompt"
python generate_replicate.py \
    "professional headshot portrait, studio lighting, sharp focus" \
    --negative "blurry, low quality, distorted, ugly" \
    --model sdxl \
    --output "$OUTPUT_DIR/example2_portrait.png"

# Example 3: Abstract art with high CFG
echo ""
echo "Example 3: Abstract art with high guidance"
python generate_replicate.py \
    "abstract geometric patterns, vibrant colors, digital art" \
    --cfg-scale 9.0 \
    --steps 40 \
    --model sdxl \
    --output "$OUTPUT_DIR/example3_abstract.png"

# Example 4: Fast generation with SD 1.5
echo ""
echo "Example 4: Fast generation with SD 1.5 (512×512)"
python generate_replicate.py \
    "a cute robot assistant, cartoon style" \
    --model sd15 \
    --width 512 \
    --height 512 \
    --output "$OUTPUT_DIR/example4_robot.png"

# Example 5: Generate 4 variations
echo ""
echo "Example 5: Generate 4 variations"
python generate_replicate.py \
    "fantasy dragon character design, detailed scales" \
    --num-outputs 4 \
    --model sdxl \
    --output "$OUTPUT_DIR/example5_dragon.png"

# Example 6: Reproducible with seed
echo ""
echo "Example 6: Reproducible generation with seed"
python generate_replicate.py \
    "futuristic city skyline at night, cyberpunk" \
    --seed 42 \
    --model sdxl \
    --output "$OUTPUT_DIR/example6_city.png"

# Example 7: Wide banner
echo ""
echo "Example 7: Wide banner format"
python generate_replicate.py \
    "panoramic ocean view with dramatic clouds" \
    --width 1536 \
    --height 512 \
    --model sdxl \
    --output "$OUTPUT_DIR/example7_banner.png"

echo ""
echo "✅ All examples generated successfully!"
echo "📁 Output directory: $OUTPUT_DIR"
echo ""
echo "💰 Cost estimate: ~$0.14 (7 SDXL images)"
echo "🆓 Free tier: $5-10 credits should cover ~200-500 images"
