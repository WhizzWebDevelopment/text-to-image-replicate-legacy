#!/usr/bin/env python3
"""
Text-to-Image Generation using Stable Diffusion (Local)
Uses Hugging Face's Diffusers library - most popular open-source solution
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import argparse
from pathlib import Path


def setup_pipeline(model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    """
    Initialize the Stable Diffusion pipeline
    
    Popular free models (stable versions):
    - runwayml/stable-diffusion-v1-5 (most stable, recommended)
    - CompVis/stable-diffusion-v1-4 (original, well-tested)
    - stabilityai/stable-diffusion-2-1 (newer, may need more VRAM)
    """
    print(f"Loading model: {model_id}")
    
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,  # Disable for faster inference
    )
    
    # Use DPM-Solver++ for faster generation
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to device
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ xformers enabled for efficient memory usage")
        except Exception:
            print("⚠ xformers not available, using standard attention")
            pipe.enable_attention_slicing()
    
    return pipe


def generate_image(
    prompt,
    pipeline,
    negative_prompt="blurry, bad quality, distorted, ugly",
    num_inference_steps=30,
    guidance_scale=7.5,
    width=512,
    height=512,
    seed=None
):
    """
    Generate an image from a text prompt
    
    Args:
        prompt: Text description of the image
        pipeline: Stable Diffusion pipeline
        negative_prompt: Things to avoid in the image
        num_inference_steps: More steps = better quality but slower (20-50)
        guidance_scale: How closely to follow the prompt (7-15)
        width, height: Image dimensions (must be multiples of 8)
        seed: Random seed for reproducibility
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    print(f"\n🎨 Generating image for: '{prompt}'")
    print(f"   Steps: {num_inference_steps}, Guidance: {guidance_scale}, Size: {width}x{height}")
    
    # Generate image
    with torch.autocast(pipeline.device.type if pipeline.device.type == "cuda" else "cpu"):
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
    
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument("prompt", type=str, help="Text prompt describing the image")
    parser.add_argument("--output", "-o", type=str, default="output.png", help="Output image path")
    parser.add_argument("--model", "-m", type=str, default="runwayml/stable-diffusion-v1-5", 
                       help="Model ID from Hugging Face")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--negative", type=str, default="blurry, bad quality, distorted",
                       help="Negative prompt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Setup pipeline
    pipe = setup_pipeline(args.model, args.device)
    
    # Generate image
    image = generate_image(
        prompt=args.prompt,
        pipeline=pipe,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed
    )
    
    # Save image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    print(f"✓ Image saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
