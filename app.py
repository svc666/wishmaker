import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

# Load pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")  # Use GPU if available, otherwise use CPU

def generate_image(prompt, guidance_scale=7.5, num_inference_steps=10):
    try:
        with torch.no_grad():
            image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
            return image
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

st.title("Stable Diffusion Image Generator")

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    if prompt:
        st.write(f"Generating image for prompt: {prompt}")
        image = generate_image(prompt)
        if image:
            st.image(image, caption='Generated Image')
    else:
        st.warning("Please enter a prompt.")
