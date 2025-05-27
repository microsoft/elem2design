import os
import base64
from mimetypes import guess_type
from tqdm import tqdm
import json
import datasets
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import time


MAX_RETRY = 5
ROLE_LIST = ["Background", "Underlay", "Logo/Image", "Embellishment", "Logo", "Image"]
TASK_PROMPT = (
    "You are an excellent graphic designer and capable of identifying the role of each element within a complete design. "
    "Your task is to determine the role of the given element, which is rendered as an image. "
    "There are 4 possible options: `Background`, `Underlay`, `Logo/Image` or `Embellishment`. "
    "Please refer to the detailed descriptions below to make your prediction. "
    "\nBackground: The foundational layer of the design, typically large in size and covering the entire canvas. It may consist of a solid color, gradient, landscape image, or similar visual foundation. "
    "\nUnderlay: A supportive layer placed beneath key content, often used to create contrast or highlight the main design elements, such as borders, buttons, color overlays, and so on. "
    "\nLogo/Image: A core visual element that represents a brand, product, or entity. It combines both imagery and logo elements to capture attention and convey the primary message. "
    "\nEmbellishment: Decorative elements that enhance visual appeal without conveying core information. These elements add style to the design. Note that they are usually small in size. "
    "\nWhen you respond, please output only one word from the 4 options. "
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


def load_crello():
    crello = datasets.load_dataset("cyberagent/crello", revision="4.0.0")
    crello_test = crello["test"]
    features = crello_test.features
    return crello_test, features


def get_response(image_path, max_retry=MAX_RETRY, max_tokens=4000):
    image = image_to_data_url(image_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistent. "}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TASK_PROMPT},
                {"type": "text", "text": "The element is: "},
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": "Please predict the given element role: "},
            ],
        },
    ]
    answer = ""
    retry_cnt = 0
    while not answer and retry_cnt < max_retry:
        try:
            time.sleep(0.5)
            # call gpt4
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
    save_dir = "./output"
    os.makedirs(save_dir, exist_ok=True)
    errorlog = open("./error.log", "a")

    for data_item in tqdm(crello):
        idx = data_item["id"]
        save_path = os.path.join(save_dir, idx)
        os.makedirs(save_path, exist_ok=True)
        length = data_item["length"]

        ele_type = data_item["type"]
        images = data_item["image"]
        texts = data_item["text"]

        data = {"id": idx, "length": length, "ele_id": None, "role": None}

        for ele_id in range(length):
            if os.path.exists(os.path.join(save_path, f"{ele_id}.json")):
                continue
            if ele_type[ele_id] != 1 or texts[ele_id] == "":
                prompt_image = images[ele_id]
                role = get_response(prompt_image)
            else:
                role = "Text"

            if role != "":
                data["ele_id"] = ele_id
                data["role"] = role
                with open(os.path.join(save_path, f"{ele_id}.json"), "w") as f:
                    json.dump(data, f, indent=4)
            else:
                errorlog.write(f"design {idx}, element {ele_id}, empty result! \n")
