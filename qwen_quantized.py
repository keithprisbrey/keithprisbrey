import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from auto_gptq import AutoGPTQForCausalLM
import sys
import os

def analyze_image_with_quantized_qwen(image_path: str, prompt: str = "What do you see in this image?") -> str:
    """
    Analyze a JPEG image using quantized Qwen2-VL model (GGUF/GPTQ) on Mac Mini.
    
    Quantized models are:
    - 4-8x smaller than full models
    - Faster inference
    - Much lower memory usage
    - Perfect for Mac Mini
    
    Args:
        image_path: Path to the JPEG image file
        prompt: The question or prompt for the model
    
    Returns:
        The model's analysis of the image
    """
    
    # Check if image file exists
    if not image_path.endswith(('.jpg', '.jpeg', '.png')):
        raise ValueError("Please provide a valid image file (JPG, JPEG, or PNG)")
    
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"✓ Image loaded: {image_path}")
        print(f"  Size: {image.size}")
        
        # Use CPU for Mac Mini
        device = "cpu"
        
        print(f"✓ Using device: {device}")
        
        # Option 1: GPTQ Quantized Model (recommended for Mac)
        # These models are already quantized and optimized
        model_name = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"
        print(f"✓ Loading quantized model: {model_name}")
        
        # Load quantized model
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            device=device,
            use_triton=False,  # Triton not available on Mac
            use_safetensors=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        print("✓ Quantized model loaded successfully")
        print(f"  Model size: ~600MB (vs ~10GB for full model)")
        
        # Prepare the message for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        
        # Process the image and prompt
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
        
        # Generate response with quantized model
        print("⏳ Generating response (quantized inference)...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
            )
        
        # Decode the response
        response = processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def analyze_with_gguf_ollama(image_path: str, prompt: str = "What do you see in this image?") -> str:
    """
    Alternative: Use Ollama with GGUF quantized models (simpler, lighter).
    Requires Ollama to be installed and running.
    
    This is the EASIEST approach for Mac Mini:
    - Install: brew install ollama
    - Pull model: ollama pull qwen2-vision
    - Run: python script.py image.jpg "your prompt"
    """
    import subprocess
    import json
    
    try:
        # Check if Ollama is running
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            raise Exception("Ollama is not running. Start it with: ollama serve")
        
        print("✓ Ollama is running")
        
        # Read image as base64
        import base64
        with open(image_path, "rb") as img_file:
            image_data = base64.standard_b64encode(img_file.read()).decode()
        
        print(f"✓ Image loaded: {image_path}")
        print(f"✓ Processing with Ollama (GGUF quantized model)")
        
        # Call Ollama API
        result = subprocess.run(
            [
                "ollama",
                "run",
                "qwen2-vision",
                f"[IMG]{image_data}[/IMG] {prompt}"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            raise Exception(f"Ollama error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        raise Exception("Ollama request timed out")
    except Exception as e:
        raise Exception(f"Ollama error: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze images with quantized Qwen2-VL on Mac Mini"
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
        choices=["gptq", "ollama"],
        default="gptq",
        help="Method: 'gptq' (Transformers) or 'ollama' (GGUF, simpler)"
    )
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*60)
        print("Quantized Qwen2-VL Image Analyzer for Mac Mini")
        print("="*60 + "\n")
        
        if args.method == "ollama":
            result = analyze_with_gguf_ollama(args.image, args.prompt)
        else:
            result = analyze_image_with_quantized_qwen(args.image, args.prompt)
        
        print("\n📊 Analysis Result:")
        print("-" * 60)
        print(result)
        print("-" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nUsage:")
        print("  python qwen_quantized.py image.jpg")
        print("  python qwen_quantized.py image.jpg -p 'Your custom prompt'")
        print("  python qwen_quantized.py image.jpg --method ollama")
        print("\nFor Ollama method:")
        print("  brew install ollama")
        print("  ollama pull qwen2-vision")
        sys.exit(1)
