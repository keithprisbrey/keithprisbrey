import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import sys

def analyze_image_with_qwen(image_path: str, prompt: str = "What do you see in this image?") -> str:
    """
    Analyze a JPEG image using Qwen2-VL model optimized for Mac Mini.
    
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
        
        # Use CPU for Mac Mini (Metal Performance Shaders will be used automatically)
        device = "cpu"
        dtype = torch.float32  # Use float32 for better Mac compatibility
        
        print(f"✓ Using device: {device}")
        
        # Load the smaller Qwen2-VL model (2B parameter version)
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        print(f"✓ Loading model: {model_name}")
        
        processor = AutoProcessor.from_pretrained(model_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True
        )
        
        print("✓ Model loaded successfully")
        
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
        
        # Generate response
        print("⏳ Generating response...")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)
        
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


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 else "What do you see in this image? Describe it in detail."
    else:
        # Default example - you can modify this
        image_file = "sample.jpg"
        prompt = "What do you see in this image? Describe it in detail."
    
    try:
        print("\n" + "="*60)
        print("Qwen2-VL Image Analyzer for Mac Mini")
        print("="*60 + "\n")
        
        result = analyze_image_with_qwen(image_file, prompt)
        
        print("\n📊 Analysis Result:")
        print("-" * 60)
        print(result)
        print("-" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nUsage: python qwen_image_analyzer.py <image_path> [prompt]\n")
        sys.exit(1)
