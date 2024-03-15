#!/usr/bin/env python3
"""
Quick Test Script
Verify your installation and generate a test image
"""

import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required = {
        "torch": "PyTorch",
        "diffusers": "Diffusers",
        "transformers": "Transformers",
        "PIL": "Pillow",
        "gradio": "Gradio"
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_gpu():
    """Check if GPU is available"""
    print("\n🖥️  Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✓ GPU available: {gpu_name}")
            print(f"   ✓ VRAM: {gpu_memory:.1f} GB")
            return "cuda"
        else:
            print("   ⚠ No GPU detected - will use CPU (slower)")
            return "cpu"
    except Exception as e:
        print(f"   ⚠ Error checking GPU: {e}")
        return "cpu"


def generate_test_image():
    """Generate a simple test image (stable version)"""
    print("\n🎨 Generating test image...")
    print("   This will take a few minutes on first run (downloading model)")
    print("   Using Stable Diffusion v1.5 (current stable version)")
    
    try:
        from generate_local import setup_pipeline, generate_image
        
        # Use stable v1.5 model
        pipe = setup_pipeline("runwayml/stable-diffusion-v1-5")
        
        # Simple test prompt
        prompt = "a cute robot waving hello, cartoon style, simple"
        
        image = generate_image(
            prompt=prompt,
            pipeline=pipe,
            num_inference_steps=20,  # Faster for testing
            width=512,
            height=512,
        )
        
        # Save
        image.save("test_output.png")
        print("\n✓ Success! Test image saved as: test_output.png")
        print("   You can now use the other scripts to generate more images!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("TEXT-TO-IMAGE GENERATOR - INSTALLATION TEST")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check GPU
    device = check_gpu()
    
    # Ask user if they want to generate test image
    print("\n" + "=" * 60)
    response = input("\nGenerate a test image? (y/n): ").strip().lower()
    
    if response == 'y':
        success = generate_test_image()
        sys.exit(0 if success else 1)
    else:
        print("\n✓ Setup verified! You're ready to generate images.")
        print("\nQuick start:")
        print("  python generate_local.py 'your prompt here'")
        print("  python gradio_ui.py")


if __name__ == "__main__":
    main()
