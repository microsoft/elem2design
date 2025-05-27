import argparse
import base64
import json
import os
import time
from io import BytesIO
from mimetypes import guess_type

import datasets
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from datasets import concatenate_datasets
from openai import AzureOpenAI
from PIL import Image
from tqdm import tqdm


MAX_RETRY = 5
ROLE_LIST = ["Background", "Underlay", "Logo/Image", "Embellishment", "Logo", "Image"]
TASK_PROMPT = (
    "You are an excellent graphic designer and capable of identifying the role of each element within a complete design. "
    "I will give you a complete graphic design and one of its elements, both rendered as images. "
    "Your task is to determine the role of the given element in the overall design. "
    "There are 4 possible options: `Background`, `Underlay`, `Logo/Image`, or `Embellishment`. "
    "Please refer to the detailed descriptions below to make your prediction. "
    "Background: The foundational layer of the design, typically large in size and covering the entire canvas. It may consist of a solid color, gradient, landscape image, or similar visual foundation. "
    "Underlay: A supportive layer placed beneath key content, often used to create contrast or highlight the main design elements, such as borders, buttons, color overlays, and so on. "
    "Logo/Image: A core visual element that represents a brand, product, or entity. It combines both imagery and logo elements to capture attention and convey the primary message. "
    "Embellishment: Decorative elements that enhance visual appeal without conveying core information. These elements add style to the design. Note that they are usually small in size. "
    "Please also consider the provided canvas and element width/height, as they might be helpful in making a decision. "
    "When you respond, please output only one word from the 4 options. "
    "Do not include any additional explanations or irrelevant information. "
)


def local_image_to_data_url(image_path):
    # Encode a local image into data URL
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def image_to_data_url(image: Image.Image, mime_type: str = "image/png") -> str:
    # Encode a PIL Image into a data URL
    buffer = BytesIO()
    image.save(buffer, format=mime_type.split("/")[-1].upper())
    base64_encoded_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def setup_client():
    endpoint = ""
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "")
    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        azure_endpoint=endpoint,
        api_version="2024-02-15-preview",
    )
    return client


def load_crello():
    crello = datasets.load_dataset("cyberagent/crello", revision="4.0.0")
    cr_train = crello["train"]
    features = cr_train.features
    cr_val = crello["validation"]
    crello = concatenate_datasets([cr_train, cr_val])
    return crello, features


def get_response(
    client,
    deployment,
    image_path,
    canvas_width,
    canvas_height,
    ele_width,
    ele_height,
    max_retry=MAX_RETRY,
    max_tokens=4000,
):
    images = [image_to_data_url(path) for path in image_path]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistent. "}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TASK_PROMPT},
                {"type": "text", "text": "The overall design is: "},
                {"type": "image_url", "image_url": {"url": images[0]}},
                {
                    "type": "text",
                    "text": f"The canvas width is {canvas_width}px, canvas height is {canvas_height}px. ",
                },
                {"type": "text", "text": "The element is: "},
                {"type": "image_url", "image_url": {"url": images[1]}},
                {
                    "type": "text",
                    "text": f"The element width is {ele_width}px, element height is {ele_height}px. ",
                },
                {"type": "text", "text": "Please predict the given element role: "},
            ],
        },
    ]
    answer = ""
    retry_cnt = 0
    while not answer and retry_cnt < max_retry:
        try:
            time.sleep(0.5)
            response = client.chat.completions.create(model=deployment, messages=messages, temperature=0.2, max_tokens=max_tokens, top_p=1)
            response = response.json()
            if isinstance(response, str):
                response = json.loads(response)
            answer = response["choices"][0]["message"]["content"]
            if answer not in ROLE_LIST:
                error_info = f"wrong label {answer}"
                answer = ""
                raise RuntimeError(error_info)
        except Exception as e:
            print(f"retry: {retry_cnt}, max_retry: {max_retry}, error: {e}")
            retry_cnt += 1
    return answer


if __name__ == "__main__":
    crello, features = load_crello()
    deployment_name = "gpt-4o"
    client = setup_client()
    save_dir = "./output"
    os.makedirs(save_dir, exist_ok=True)
    errorlog = open("./error.log", "a")

    # split
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_idx", "-i", type=int)
    args = parser.parse_args()
    pid = args.patch_idx
    sharded_crello = crello.shard(num_shards=8, index=pid)
    #

    for data_item in tqdm(sharded_crello):
        idx = data_item["id"]
        save_path = os.path.join(save_dir, idx)
        os.makedirs(save_path, exist_ok=True)
        length = data_item["length"]
        canvas_width = data_item["canvas_width"]
        canvas_height = data_item["canvas_height"]
        canvas_width = int(features["canvas_width"].int2str(canvas_width))
        canvas_height = int(features["canvas_height"].int2str(canvas_height))
        ele_type = data_item["type"]
        images = data_item["image"]
        texts = data_item["text"]
        preview_image = data_item["preview"]
        widths = data_item["width"]
        heights = data_item["height"]
        widths = [int(width * canvas_width) for width in widths]
        heights = [int(height * canvas_height) for height in heights]
        data = {
            "id": idx,
            "length": length,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "ele_id": None,
            "role": None,
        }

        for ele_id in range(length):
            if os.path.exists(os.path.join(save_path, f"{ele_id}.json")):
                continue
            if ele_type[ele_id] != 1 or texts[ele_id] == "":
                prompt_image = [preview_image, images[ele_id]]
                role = get_response(
                    client,
                    deployment_name,
                    prompt_image,
                    canvas_width,
                    canvas_height,
                    widths[ele_id],
                    heights[ele_id],
                )
            else:
                role = "Text"

            if role != "":
                data["ele_id"] = ele_id
                data["role"] = role
                with open(os.path.join(save_path, f"{ele_id}.json"), "w") as f:
                    json.dump(data, f, indent=4)
            else:
                errorlog.write(f"design {idx}, element {ele_id}, empty result! \n")
