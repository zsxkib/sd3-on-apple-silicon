import os
import time
import torch
import psutil
from diffusers import StableDiffusion3Pipeline

SD3_MODEL_CACHE = "./sd3-cache"

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    cache_dir=SD3_MODEL_CACHE,
)

# Check available system RAM and enable attention slicing if less than 64 GB
if (available_ram := psutil.virtual_memory().available / (1024**3)) < 64:
    pipe.enable_attention_slicing()

# Automatically infer the device and prioritize "mps" if available
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
pipe = pipe.to(device)
print(f"Pipeline moved to device: {pipe.device}")

seed = None
if seed is None:
    seed = int.from_bytes(os.urandom(2), "big")
print(f"Using seed: {seed}")
generator = torch.Generator(device=device).manual_seed(seed)

start_time = time.time()
with torch.device(device):
    print(f"Current device: {torch.device(device)}")
    image = pipe(
        prompt=(prompt := "A cat holding a sign that says hello world"),
        height=(height := 512),
        width=(width := 512),
        num_inference_steps=28,
        guidance_scale=7.0,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
        return_dict=True,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
end_time = time.time()
print(f"Image generated successfully in {end_time - start_time:.2f} seconds.")

if image and (image_path := "sd3-output-mps.png"):
    image.save(image_path)
