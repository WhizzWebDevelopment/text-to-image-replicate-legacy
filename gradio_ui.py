#!/usr/bin/env python3
"""
Gradio Web UI for Text-to-Image Generation
Simple web interface for generating images from text prompts
"""

import gradio as gr
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np


# Global pipeline (loaded once)
pipeline = None


def load_model(model_id="runwayml/stable-diffusion-v1-5"):
    """Load the model once at startup (stable version)"""
    global pipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    
    pipeline = pipeline.to(device)
    
    if device == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pipeline.enable_attention_slicing()
    
    print("✓ Model loaded successfully!")
    return pipeline


def generate_image_ui(
    prompt,
    negative_prompt,
    num_steps,
    guidance_scale,
    width,
    height,
    seed,
    use_random_seed
):
    """
    Generate image from the UI
    """
    global pipeline
    
    if pipeline is None:
        return None, "❌ Model not loaded. Please wait..."
    
    try:
        # Handle seed
        generator = None
        if not use_random_seed and seed >= 0:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed)
            actual_seed = seed
        else:
            actual_seed = np.random.randint(0, 2**32 - 1)
            generator = torch.Generator(device=pipeline.device).manual_seed(actual_seed)
        
        # Generate
        with torch.autocast(pipeline.device.type if pipeline.device.type == "cuda" else "cpu"):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(num_steps),
                guidance_scale=guidance_scale,
                width=int(width),
                height=int(height),
                generator=generator,
            )
        
        image = result.images[0]
        info = f"✓ Generated successfully!\nSeed: {actual_seed}"
        
        return image, info
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_ui():
    """Create the Gradio interface (Gradio 3.x compatible)"""
    
    with gr.Blocks(title="Text-to-Image Generator") as demo:
        gr.Markdown("# 🎨 Text-to-Image Generator (Stable Version)")
        gr.Markdown("Generate images from text using Stable Diffusion v1.5")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a beautiful sunset over mountains, oil painting style",
                    lines=3
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="blurry, bad quality, distorted, ugly, duplicate",
                    lines=2
                )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=1,
                        label="Steps (more = better quality)"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale (how closely to follow prompt)"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Width"
                    )
                    
                    height = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Height"
                    )
                
                with gr.Row():
                    use_random_seed = gr.Checkbox(label="Random Seed", value=True)
                    seed = gr.Number(label="Seed (for reproducibility)", value=42)
                
                generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                
                # Preset examples
                gr.Markdown("### 💡 Example Prompts")
                gr.Examples(
                    examples=[
                        ["a majestic lion in a savanna at sunset, photorealistic, 4k"],
                        ["a cyberpunk city with neon lights, futuristic, detailed"],
                        ["a cozy cottage in a forest, autumn, oil painting"],
                        ["an astronaut riding a horse on mars, cinematic lighting"],
                        ["a bowl of ramen with steam, food photography, close-up"],
                    ],
                    inputs=prompt,
                )
            
            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="Generated Image", type="pil")
                output_info = gr.Textbox(label="Info", lines=2)
        
        # Connect button
        generate_btn.click(
            fn=generate_image_ui,
            inputs=[
                prompt, negative_prompt, num_steps, guidance_scale,
                width, height, seed, use_random_seed
            ],
            outputs=[output_image, output_info]
        )
    
    return demo


def main():
    # Load model
    print("Loading Stable Diffusion model...")
    load_model()
    
    # Create and launch UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create a public link
    )


if __name__ == "__main__":
    main()
