from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import IPython.display as display

# Load pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")  # Use GPU if available, otherwise use CPU

# Function to generate image from text prompt
def generate_image(prompt, guidance_scale=7.5, num_inference_steps=50):
    try:
        with torch.no_grad():
            # Generate image with refined prompt handling
            image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
            return image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Define a detailed prompt
prompt = "ganesh chavati wishs to everyone "

# Generate and display the image
print(f"Generating image for prompt: {prompt}")
image = generate_image(prompt)
if image:
    display.display(image)
