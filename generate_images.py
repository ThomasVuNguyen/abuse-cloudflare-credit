import os
import re
import random
import requests
from pathlib import Path
from dotenv import load_dotenv

def sanitize_filename(text):
    """Convert text to a safe filename."""
    # Remove invalid characters and replace spaces with underscores
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[\s]+', '_', text)
    # Limit length
    return text[:50]  # Limit to 50 chars to avoid too long filenames

def generate_image_from_text(
    prompt,
    model_name="@cf/stabilityai/stable-diffusion-xl-base-1.0",
    negative_prompt="",
    height=1024,
    width=1024,
    num_steps=20,
    guidance=7.5,
    seed=None,
    output_dir="generated_images"
):
    # Load environment variables from .env file
    load_dotenv()
    
    # Get Cloudflare credentials from environment variables
    account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
    api_token = os.getenv('CLOUDFLARE_API_TOKEN')
    
    if not account_id or not api_token:
        raise ValueError("Missing required environment variables. Please check your .env file.")
    
    # Set up the Cloudflare Workers AI endpoint
    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
    
    # Create output directory if it doesn't exist
    model_dir = Path(output_dir) / model_name.replace('@', '').replace('/', '_')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename from prompt and random number
    safe_prompt = sanitize_filename(prompt)
    random_suffix = random.randint(1000, 9999)
    output_filename = f"{safe_prompt}_{random_suffix}.png"
    output_path = model_dir / output_filename
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_steps": num_steps,
        "guidance": guidance
    }
    if seed is not None:
        payload["seed"] = seed

    # Print debug info
    print(f"Sending request to: {endpoint}")
    print(f"Using payload: {payload}")
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload, stream=True)
        
        # Print response headers for debugging
        print("\nResponse headers:")
        for key, value in response.headers.items():
            print(f"{key}: {value}")
            
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            print(f"\nContent-Type: {content_type}")
            
            # Check if the response is an image
            if 'image/' in content_type:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"\n✅ Image successfully saved to {output_path.absolute()}")
                return str(output_path.absolute())
            else:
                print("\n❌ Unexpected response content type. Response content:")
                print(response.text[:1000])  # Print first 1000 chars to avoid huge outputs
        else:
            print(f"\n❌ Failed to generate image. Status code: {response.status_code}")
            print("Response content:", response.text)
            
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
    
    return False

if __name__ == "__main__":
    # ===== CONFIGURATION =====
    # Change these values as needed
    MODEL_NAME = "@cf/stabilityai/stable-diffusion-xl-base-1.0"  # Try other models like "@cf/lykon/dreamshaper-8-lcm"
    PROMPT = "A futuristic cityscape at sunset, ultra-detailed, vibrant colors"
    NUM_IMAGES = 1  # Number of images to generate
    # =========================
    
    print(f"Generating {NUM_IMAGES} image(s) with model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}")
    
    # Generate the specified number of images
    for i in range(NUM_IMAGES):
        print(f"\n--- Generating image {i + 1}/{NUM_IMAGES} ---")
        result = generate_image_from_text(
            prompt=PROMPT,
            model_name=MODEL_NAME,
            output_dir="generated_images"
        )
        if result:
            print(f"✅ Success: {result}")
        else:
            print("❌ Failed to generate image")

if __name__ == "__main__":
    main()
