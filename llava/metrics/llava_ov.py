import argparse
import json
import os
from glob import glob

from PIL import Image

import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, required=True)
args = parser.parse_args()
folder = args.input

PROMPT = "You are an autonomous AI assistant that assists designers by providing insightful, objective, and constructive critiques on graphic design projects. Your goal is to: Provide a thorough and unbiased evaluation of graphic designs based on established design principles and industry standards. Maintain consistent and high standards of critique. Please follow the rules below: Strive to grade as objectively as possible. Grade carefully. Designs with a blank canvas will receive 0 points. Reply in JSON format only, no additional information is required. Scoring criteria: Design and Layout (1-10): Graphic designs should present a clean, balanced, and consistent layout. The organization of elements should enhance the message and provide a clear path for the eye. A score of 10 indicates that the layout maximizes readability and visual appeal, while a score of 1 indicates that the layout is cluttered and has no clear hierarchy or flow. Content Relevance and Effectiveness (1-10): Content should not only be relevant to its purpose, but also engage the target audience and effectively communicate the intended message. A score of 10 indicates that the content resonates with the target audience, fits the purpose of the design, and enhances the overall message. A score of 1 indicates that the content is irrelevant or does not connect with the audience. Typography and Color Scheme (1-10): Typography and color should work together to enhance readability and coordinate with other design elements. This includes font choice, size, line spacing, color and placement, as well as the overall color scheme of the design. A score of 10 indicates that typography and color are used appropriately and fit the purpose and aesthetic of the design, while a score of 1 indicates that these elements are used inappropriately, hinder readability, or clash with the design. Graphics and Imagery (1-10): Any graphics or images used should enhance the design, not distract from it. They should be high quality, relevant, and coordinate with the other elements. A score of 10 indicates that the graphic or image enhances the overall design and message, while a score of 1 indicates that the visual is low quality, irrelevant, or distracts. Innovation and Originality (1-10): The design should demonstrate originality and creativity. It should not only follow trends, but also show a unique interpretation of the brief. A score of 10 indicates that the design is highly creative and innovative, with its originality standing out, while a score of 1 indicates a lack of creativity or a cookie-cutter approach. Be cautious about giving high scores unless they are truly excellent."

model_id = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

fp = open(os.path.join(folder, "score.jsonl"), "w")
images = glob(os.path.join(folder, "render/*_4.png"))

for image in tqdm(images):
    image = Image.open(image)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
    output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    res = processor.decode(output[0][2:], skip_special_tokens=True)
    fp.write(json.dumps({"res": res}) + "\n")
