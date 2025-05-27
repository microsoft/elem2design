# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import IGNORE_INDEX
from llava.conversation import layout_conv
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model import *
from llava.model.builder import load_pretrained_model

BASE_MODEL = {
    'Llama-3.1-8B': 'meta-llama/Llama-3.1-8B',
    'llama3-llava-next-8b': 'lmms-lab/llama3-llava-next-8b',
    'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'llava-v1.5-7b': 'liuhaotian/llava-v1.5-7b',
    "Mistral-7B-v0.3": "mistralai/Mistral-7B-v0.3",
}


def white_rgb_convert(img: Image):
    img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img)
    return bg


@dataclass
class EvalArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    output_dir: str = field(default='./')
    first_n: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    num_return: int = 1


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: EvalArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        if data_args.first_n is not None:
            list_data_dict = list_data_dict[:data_args.first_n]
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        image_list = sources['image']
        processor = self.data_args.image_processor

        images = []
        for image_file in image_list:
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            try:
                image = Image.open(os.path.join(self.data_args.image_folder, image_file))
                image = white_rgb_convert(image)
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt', input_data_format="channels_last")['pixel_values'][0]
            except:
                image = torch.zeros(3, 336, 336)
            images.append(image)

        conv = layout_conv.copy()
        conv.sep2 = self.tokenizer.eos_token
        conversations = []
        for sentence in sources['conversations']:
            if sentence['from'] == 'gpt':
                sentence['value'] = None
            conv.append_message(sentence['from'], sentence['value'])
            if sentence['from'] == 'gpt':
                break
        conversations.append(conv.get_prompt())

        input_ids = torch.stack([tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        data_dict = dict(input_ids=input_ids[0])

        if len(images) == 0:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
        else:
            data_dict['image'] = torch.stack(images, dim=0)
        data_dict['id'] = sources['id']
        data_dict['render_image'] = sources['render_image']
        data_dict['render_text'] = sources['render_text']
        data_dict['conversations'] = sources['conversations']
        data_dict['canvas_width'] = int(re.findall(r'width (\d+)px', sources['conversations'][0]['value'])[0])
        data_dict['canvas_height'] = int(re.findall(r'height (\d+)px', sources['conversations'][0]['value'])[0])
        return data_dict


def run_eval():
    parser = transformers.HfArgumentParser(EvalArguments)
    eval_args, = parser.parse_args_into_dataclasses()
    model_name = get_model_name_from_path(eval_args.model_name_or_path)
    model_path = eval_args.model_name_or_path
    if 'lora' not in model_path:
        model_base = None
    else:
        import re
        model_base = re.findall('\/([^\/]+)_lora', model_path)[0]
        model_base = BASE_MODEL[model_base]
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name,
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id or 0
    eval_args.image_processor = image_processor

    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=eval_args.data_path,
        data_args=eval_args
    )
    model = model.to('cuda')
    if not os.path.exists(eval_args.output_dir):
        os.makedirs(eval_args.output_dir, exist_ok=True)
    f = open(os.path.join(eval_args.output_dir, 'pred.jsonl'), 'w')

    with torch.inference_mode():
        for idx in tqdm(range(len(eval_dataset))):
            sample = eval_dataset[idx]
            data = {
                'num': idx,
                'id': sample['id'],
                'render_image': sample['render_image'],
                'render_text': sample['render_text'],
                'predictions': [''],
                'canvas_width': sample['canvas_width'],
                'canvas_height': sample['canvas_height'],
            }
            input_ids = sample['input_ids'].unsqueeze(0).to('cuda')
            images = [sample['image'].to('cuda', dtype=torch.float16)]
            attention_mask=input_ids.ne(tokenizer.pad_token_id).to('cuda')

            try:
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=eval_args.temperature,
                    top_p=eval_args.top_p,
                    num_return_sequences=eval_args.num_return,
                    pad_token_id=tokenizer.eos_token_id,
                    max_length=5000
                )
                output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                data['predictions'] = [item + ' ##### ' + output + ' $$$$$ ' for item in data['predictions']]
            except Exception as e:
                print(e, idx)

            f.write(json.dumps(data) + "\n")
        

if __name__ == "__main__":
    run_eval()
