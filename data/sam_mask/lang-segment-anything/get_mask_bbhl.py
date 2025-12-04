# https://github.com/luca-medeiros/lang-segment-anything

from PIL import Image
from lang_sam import LangSAM
import numpy as np
import os


# Function to apply transparent masks on the image
def save_mask_results(image, masks, output_path):
    # Ensure the image is in RGBA mode (with transparency support)
    image = image_pil.convert("RGBA")

    # Iterate over the binary masks
    for mask in masks:
        # Convert mask to the same size as the image
        mask_resized = Image.fromarray(mask).resize(image.size, Image.NEAREST).convert("L")
        
        # Create an RGBA mask with the same size
        mask_rgba = mask_resized.convert("RGBA")
        
        # Get the alpha channel from the mask to apply transparency
        mask_rgba = np.array(mask_rgba)
        alpha_channel = mask_rgba[:, :, 0]  # Assuming the mask is single-channel
        
        # Create a new transparent image with the same size as the input image
        transparent_overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        
        # Set the red, green, and blue channels to some color (e.g., red)
        transparent_overlay[:, :, 0] = 255  # Red channel
        transparent_overlay[:, :, 1] = 0    # Green channel
        transparent_overlay[:, :, 2] = 0    # Blue channel
        
        # Set the alpha channel to apply transparency from the mask
        transparent_overlay[:, :, 3] = alpha_channel
        
        # Convert the overlay to an Image object
        overlay_image = Image.fromarray(transparent_overlay, "RGBA")
        
        # Composite the overlay with the original image
        image = Image.alpha_composite(image, overlay_image)
    
    # Save the resulting image
    image.save(output_path, "PNG")

# Function to save a binary mask as an image
def save_binary_mask(mask, output_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert the binary mask to a PIL Image object
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Multiply by 255 to make 1 white and 0 black
    # Save the mask as an image
    mask_image.save(output_path)


model = LangSAM()
image_paths = [
    f"/home/lemonhdl/workspace/DemoGen/data/sam_mask/beat_block_hammer_loop5/000000.jpg"
    for i in range(1)
]
# text_prompt = "an iron hammer with black handle"
text_prompt = "a red cubic building block"

for idx, image_path in enumerate(image_paths):
    image_pil = Image.open(image_path).convert("RGB")
    results = model.predict([image_pil], [text_prompt])
    masks = results[0]["masks"]
    print(masks.shape)
    # assert masks.shape[0] == 1
    save_binary_mask(
        masks[0],
        f"/home/lemonhdl/workspace/DemoGen/data/sam_mask/beat_block_hammer_loop5/0/block.jpg"
    )