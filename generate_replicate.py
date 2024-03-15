#!/usr/bin/env python3
"""
Text-to-Image Generation using Replicate API

Replicate is a commercial service with free tier (free tier available)
Uses Stable Diffusion XL via simple API
"""

import os
import sys
import argparse
import replicate
from PIL import Image
import requests
from io import BytesIO
from typing import Optional, List


class ReplicateImageGenerator:
    """
    Replicate API for Stable Diffusion XL image generation
    
    Free tier: $5-10 in free credits (free tier available)
    No GPU required - runs in cloud
    """
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize Replicate client
        
        Args:
            api_token: Replicate API token (or use REPLICATE_API_TOKEN env var)
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            raise ValueError(
                "Replicate API token required!\n"
                "Set REPLICATE_API_TOKEN environment variable or pass api_token parameter.\n"
                "Get your token at: https://replicate.com/account/api-tokens"
            )
        
        # Set token for replicate client
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        
        print(f"✓ Replicate API initialized")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_outputs: int = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        model: str = "sdxl"
    ) -> List[Image.Image]:
        """
        Generate image(s) using Replicate API
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            width: Image width (128-1536, default 1024)
            height: Image height (128-1536, default 1024)
            num_outputs: Number of images to generate (1-4)
            num_inference_steps: Quality vs speed (1-500, default 30)
            guidance_scale: Prompt adherence (1-20, default 7.5)
            seed: Random seed for reproducibility
            model: Model to use ("sdxl" or "sd15")
        
        Returns:
            List of PIL Image objects
        """
        # Select model version
        if model == "sdxl":
            model_version = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
        elif model == "sd15":
            model_version = "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
        else:
            raise ValueError(f"Unknown model: {model}. Use 'sdxl' or 'sd15'")
        
        # Build input parameters
        input_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_outputs": num_outputs,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        
        if negative_prompt:
            input_params["negative_prompt"] = negative_prompt
        
        if seed is not None:
            input_params["seed"] = seed
        
        print(f"\n🎨 Generating with Replicate ({model.upper()})...")
        print(f"📝 Prompt: {prompt}")
        if negative_prompt:
            print(f"🚫 Negative: {negative_prompt}")
        print(f"📐 Size: {width}×{height}")
        print(f"⚙️  Steps: {num_inference_steps}, CFG: {guidance_scale}")
        if seed:
            print(f"🎲 Seed: {seed}")
        
        try:
            # Run the model
            output = replicate.run(
                model_version,
                input=input_params
            )
            
            # Output is a list of URLs
            images = []
            for i, url in enumerate(output):
                print(f"📥 Downloading image {i+1}/{len(output)}...")
                response = requests.get(url)
                response.raise_for_status()
                
                image = Image.open(BytesIO(response.content))
                images.append(image)
                print(f"✓ Image {i+1} ready: {image.size}")
            
            return images
            
        except replicate.exceptions.ReplicateError as e:
            print(f"❌ Replicate API error: {e}")
            raise
        except requests.RequestException as e:
            print(f"❌ Download error: {e}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise
    
    def list_models(self):
        """List available models and their info"""
        models = {
            "sdxl": {
                "name": "Stable Diffusion XL",
                "description": "High-quality 1024×1024 images",
                "best_for": "General purpose, high quality",
                "resolution": "1024×1024 native"
            },
            "sd15": {
                "name": "Stable Diffusion 1.5",
                "description": "Fast, standard SD model",
                "best_for": "Speed, lower resolution",
                "resolution": "512×512 native"
            }
        }
        
        print("\n📋 Available Models:\n")
        for key, info in models.items():
            print(f"  {key}: {info['name']}")
            print(f"    {info['description']}")
            print(f"    Best for: {info['best_for']}")
            print(f"    Resolution: {info['resolution']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Replicate API (free tier available)"
    )
    
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=None,
        help="Text prompt describing the desired image"
    )
    
    parser.add_argument(
        "--negative",
        type=str,
        default=None,
        help="Negative prompt (things to avoid)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (128-1536, default: 1024)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels (128-1536, default: 1024)"
    )
    
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1,
        help="Number of images to generate (1-4, default: 1)"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (1-500, default: 30)"
    )
    
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.5,
        help="Guidance scale - prompt adherence (1-20, default: 7.5)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl",
        choices=["sdxl", "sd15"],
        help="Model to use: sdxl (default) or sd15"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="replicate_output.png",
        help="Output filename (default: replicate_output.png)"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    try:
        generator = ReplicateImageGenerator()
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # List models if requested
    if args.list_models:
        generator.list_models()
        sys.exit(0)
    
    # Check for prompt
    if not args.prompt:
        print("❌ Error: Prompt is required!")
        print("\nUsage:")
        print('  python generate_replicate.py "a beautiful sunset"')
        print("\nFor more options:")
        print("  python generate_replicate.py --help")
        sys.exit(1)
    
    # Generate image
    try:
        images = generator.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            num_outputs=args.num_outputs,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg_scale,
            seed=args.seed,
            model=args.model
        )
        
        # Save images
        if len(images) == 1:
            images[0].save(args.output)
            print(f"\n✅ Image saved: {args.output}")
        else:
            # Multiple images - save with numbers
            base_name = args.output.rsplit(".", 1)[0]
            ext = args.output.rsplit(".", 1)[1] if "." in args.output else "png"
            
            for i, image in enumerate(images, 1):
                filename = f"{base_name}_{i}.{ext}"
                image.save(filename)
                print(f"✅ Image {i} saved: {filename}")
        
        print(f"\n🎉 Generation complete!")
        print(f"📊 Generated {len(images)} image(s)")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
