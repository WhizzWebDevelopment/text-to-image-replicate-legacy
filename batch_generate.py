#!/usr/bin/env python3
"""
Batch Text-to-Image Generation
Generate multiple images from a list of prompts
"""

import argparse
from pathlib import Path
from generate_local import setup_pipeline, generate_image
import json
from datetime import datetime


def load_prompts(prompt_file):
    """Load prompts from file (JSON or text)"""
    path = Path(prompt_file)
    
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "prompts" in data:
                return data["prompts"]
    else:
        # Plain text file, one prompt per line
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]


def batch_generate(
    prompts,
    output_dir="outputs",
    model_id="runwayml/stable-diffusion-v1-5",
    **kwargs
):
    """
    Generate images for a batch of prompts (stable version)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup pipeline once
    print(f"Loading model: {model_id}")
    pipe = setup_pipeline(model_id)
    
    # Generate timestamp for this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = output_dir / timestamp
    batch_dir.mkdir(exist_ok=True)
    
    results = []
    
    print(f"\n🎨 Generating {len(prompts)} images...")
    print(f"   Output directory: {batch_dir}")
    
    for i, prompt in enumerate(prompts, 1):
        try:
            print(f"\n[{i}/{len(prompts)}] {prompt[:60]}...")
            
            # Generate image
            image = generate_image(
                prompt=prompt,
                pipeline=pipe,
                **kwargs
            )
            
            # Save with sanitized filename
            safe_name = "".join(c for c in prompt[:50] if c.isalnum() or c in " -_")
            safe_name = safe_name.strip().replace(" ", "_")
            filename = f"{i:03d}_{safe_name}.png"
            
            output_path = batch_dir / filename
            image.save(output_path)
            
            results.append({
                "index": i,
                "prompt": prompt,
                "filename": filename,
                "success": True
            })
            
            print(f"   ✓ Saved: {filename}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "index": i,
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model": model_id,
        "parameters": kwargs,
        "results": results,
        "summary": {
            "total": len(prompts),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"])
        }
    }
    
    metadata_path = batch_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Batch complete!")
    print(f"   Success: {metadata['summary']['successful']}/{metadata['summary']['total']}")
    print(f"   Output: {batch_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch generate images from prompts")
    parser.add_argument("prompts", type=str, help="Path to prompts file (JSON or text)")
    parser.add_argument("--output", "-o", type=str, default="outputs", 
                       help="Output directory")
    parser.add_argument("--model", "-m", type=str, 
                       default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--negative", type=str, 
                       default="blurry, bad quality, distorted")
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    # Generate
    batch_generate(
        prompts=prompts,
        output_dir=args.output,
        model_id=args.model,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
