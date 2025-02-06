import os
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()


def classify_and_copy_images(image_folder_path):
    # Get the current working directory
    current_directory = os.getcwd()

    # Create "Raindrop" and "Non-Raindrop" folders in the current directory
    raindrop_folder = Path(current_directory) / "Raindrop"
    non_raindrop_folder = Path(current_directory) / "Non-Raindrop"
    raindrop_folder.mkdir(exist_ok=True)
    non_raindrop_folder.mkdir(exist_ok=True)

    # Iterate over each image in the folder
    for image_filename in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_filename)

        # Check if the file is a valid image (you can add more extensions if needed)
        if image_filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # Prepare the conversation and image input for the model
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nAnalyze this image carefully for any signs of rainfall\n1. Look for water droplets or streaks in the air that indicate falling rain\n2. Check for water splashes or droplets bouncing off surfaces\n4. Look for circular water marks or splash patterns on the ground\n Answer ONLY with 'Yes' or 'No' based on whether you can see ANY of these signs of rainfall.",
                    "images": [image_path],
                },
                {"role": "Assistant", "content": ""},
            ]

            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(vl_gpt.device)

            # Run the model to get the response
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

            # Get the model's answer
            answer = tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )
            print(f"Classifying {image_filename}: {answer}")
            # Copy image to the corresponding folder based on the model's response
            if answer.strip().lower() == "yes.":
                shutil.copy(image_path, raindrop_folder / image_filename)
            else:
                shutil.copy(image_path, non_raindrop_folder / image_filename)


# Specify the folder path containing the images (can be relative to the current directory)
image_folder_path = "/home/shpark/4type_weather_driving_dataset/Validation/Rainy"

# Run the classification function
classify_and_copy_images(image_folder_path)
