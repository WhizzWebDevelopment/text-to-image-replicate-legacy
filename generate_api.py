#!/usr/bin/env python3
"""
Text-to-Image Generation using Free APIs
Alternative approaches using cloud services with free tiers
"""

import os
import requests
from pathlib import Path
import argparse
from PIL import Image
from io import BytesIO


class ReplicateGenerator:
    """
    Replicate API - Free tier available (stable version)
    https://replicate.com/stability-ai/stable-diffusion
    """
    def __init__(self, api_token=None):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable not set")
        
        try:
            import replicate
            self.replicate = replicate
            # Use older API style 
            self.replicate_client = replicate.Client(api_token=self.api_token)
        except ImportError:
            raise ImportError("Install replicate: pip install replicate==0.15.4")
    
    def generate(self, prompt, **kwargs):
        """Generate image using Replicate API (Stable Diffusion v1.5)"""
        # Use SD 1.5 model - most stable 
        model = "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
        
        output = self.replicate_client.run(
            model,
            input={
                "prompt": prompt,
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "num_inference_steps": kwargs.get("num_inference_steps", 50),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 512),
            }
        )
        
        # Download image
        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content))


class HuggingFaceGenerator:
    """
    Hugging Face Inference API - Free tier available (stable)
    https://huggingface.co/docs/api-inference/
    """
    def __init__(self, api_token=None):
        self.api_token = api_token or os.getenv("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError("HF_API_TOKEN environment variable not set")
        
        # Use SD 1.5 - most stable model 
        self.api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def generate(self, prompt, **kwargs):
        """Generate image using Hugging Face Inference API"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "num_inference_steps": kwargs.get("num_inference_steps", 30),
                "guidance_scale": kwargs.get("guidance_scale", 7.5),
            }
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        return Image.open(BytesIO(response.content))


class StabilityAIGenerator:
    """
    Stability AI API - Free credits available on signup 
    https://platform.stability.ai/
    """
    def __init__(self, api_token=None):
        self.api_token = api_token or os.getenv("STABILITY_API_KEY")
        if not self.api_token:
            raise ValueError("STABILITY_API_KEY environment variable not set")
        
        # Use SD 1.5 endpoint - stable 
        self.api_url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-5/text-to-image"
    
    def generate(self, prompt, **kwargs):
        """Generate image using Stability AI API"""
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            json={
                "text_prompts": [
                    {"text": prompt, "weight": 1},
                    {"text": kwargs.get("negative_prompt", ""), "weight": -1}
                ],
                "cfg_scale": kwargs.get("guidance_scale", 7.5),
                "height": kwargs.get("height", 512),
                "width": kwargs.get("width", 512),
                "steps": kwargs.get("num_inference_steps", 30),
                "samples": 1,
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        data = response.json()
        import base64
        image_data = base64.b64decode(data["artifacts"][0]["base64"])
        return Image.open(BytesIO(image_data))


def main():
    parser = argparse.ArgumentParser(description="Generate images using free APIs")
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument("--output", "-o", type=str, default="output_api.png")
    parser.add_argument("--service", "-s", type=str, 
                       choices=["replicate", "huggingface", "stability"],
                       default="huggingface",
                       help="API service to use")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--negative", type=str, default="blurry, bad quality")
    
    args = parser.parse_args()
    
    # Select generator
    generators = {
        "replicate": ReplicateGenerator,
        "huggingface": HuggingFaceGenerator,
        "stability": StabilityAIGenerator,
    }
    
    print(f"🌐 Using {args.service} API")
    
    try:
        generator = generators[args.service]()
        
        # Generate image
        print(f"🎨 Generating: '{args.prompt}'")
        image = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
        )
        
        # Save
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        
        print(f"✓ Image saved to: {output_path.absolute()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
