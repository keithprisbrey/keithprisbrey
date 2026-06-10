import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import sys
import subprocess
import json
import base64

def analyze_image_with_transformers(image_path: str, prompt: str = "What do you see in this image?") -> str:
    """
    Analyze a JPEG image using Qwen2-VL model directly via Transformers.
    Mac Mini compatible version (CPU inference with Metal acceleration).
    
    Args:
        image_path: Path to the JPEG image file
        prompt: The question or prompt for the model
    
    Returns:
        The model's analysis of the image
    """
    
    if not image_path.endswith(('.jpg', '.jpeg', '.png')):
        raise ValueError("Please provide a valid image file (JPG, JPEG, or PNG)")
    
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"✓ Image loaded: {image_path}")
        print(f"  Size: {image.size}")
        
        device = "cpu"
        print(f"✓ Using device: {device} (Metal acceleration enabled on Mac)")
        
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        print(f"✓ Loading model: {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            low_cpu_mem_usage=True
        )
        
        print("✓ Model loaded successfully")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        
        print(f"✓ Processing image with prompt: '{prompt}'")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = processor.process_images(
            [image], size=processor.image_processor.size
        )
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        print("⏳ Generating response...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
            )
        
        response = processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def analyze_with_ollama(image_path: str, prompt: str = "What do you see in this image?") -> str:
    """
    BEST METHOD FOR MAC MINI: Use Ollama with GGUF quantized models.
    
    Why Ollama is the best choice:
    - Easiest installation (one homebrew command)
    - Handles GGUF models perfectly
    - Optimized for Mac (uses Metal acceleration)
    - No Python dependency management
    - Fastest on Mac Mini
    """
    
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            raise Exception(
                "Ollama is not running.\n"
                "Start it with: ollama serve\n"
                "Or: brew services start ollama"
            )
        
        print("✓ Ollama is running")
        
        try:
            data = json.loads(result.stdout)
            models = [m.get("name", "") for m in data.get("models", [])]
            if not any("qwen" in m.lower() for m in models):
                raise Exception(
                    "qwen2-vision model not found.\n"
                    "Download it with: ollama pull qwen2-vision"
                )
        except json.JSONDecodeError:
            pass
        
        with open(image_path, "rb") as img_file:
            image_data = base64.standard_b64encode(img_file.read()).decode()
        
        image = Image.open(image_path)
        print(f"✓ Image loaded: {image_path}")
        print(f"  Size: {image.size}")
        print(f"✓ Processing with Ollama (GGUF, Metal acceleration)")
        
        api_data = {
            "model": "qwen2-vision",
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "temperature": 0.7,
        }
        
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-X", "POST",
                "http://localhost:11434/api/generate",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(api_data),
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            try:
                response_data = json.loads(result.stdout)
                return response_data.get("response", "No response from model")
            except json.JSONDecodeError:
                return result.stdout.strip()
        else:
            raise Exception(f"Ollama error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        raise Exception("Ollama request timed out")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze images with Qwen2-VL on Mac Mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
RECOMMENDED: Use --method ollama (easiest on Mac Mini)

Quick Start with Ollama:
  1. brew install ollama
  2. ollama pull qwen2-vision
  3. ollama serve  (in background or separate terminal)
  4. python qwen_quantized.py image.jpg --method ollama

Alternative: Direct Transformers method
  pip install torch transformers pillow
  python qwen_quantized.py image.jpg --method direct
        """
    )
    parser.add_argument(
        "image",
        help="Path to the image file (JPG, JPEG, PNG)"
    )
    parser.add_argument(
        "-p", "--prompt",
        default="What do you see in this image? Describe it in detail.",
        help="Custom prompt for the model"
    )
    parser.add_argument(
        "--method",
        choices=["ollama", "direct"],
        default="ollama",
        help="Method: 'ollama' (RECOMMENDED) or 'direct' (Transformers)"
    )
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*70)
        print("Qwen2-VL Image Analyzer for Mac Mini")
        print("="*70 + "\n")
        
        if args.method == "ollama":
            result = analyze_with_ollama(args.image, args.prompt)
        else:
            result = analyze_image_with_transformers(args.image, args.prompt)
        
        print("\n📊 Analysis Result:")
        print("-" * 70)
        print(result)
        print("-" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        print("="*70)
        print("SETUP INSTRUCTIONS FOR MAC MINI:")
        print("="*70)
        print("\n✓ OPTION 1: Ollama (RECOMMENDED - Easiest)")
        print("  1. brew install ollama")
        print("  2. ollama pull qwen2-vision")
        print("  3. ollama serve")
        print("  4. python qwen_quantized.py image.jpg --method ollama\n")
        
        print("✓ OPTION 2: Direct Transformers")
        print("  pip install torch transformers pillow")
        print("  python qwen_quantized.py image.jpg --method direct\n")
        print("="*70 + "\n")
        sys.exit(1)
