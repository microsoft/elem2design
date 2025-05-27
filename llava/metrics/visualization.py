import argparse
import json
from pathlib import Path

from common.io import read_json, read_jsonl
from crello.util import render
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str)
    parser.add_argument("--image_base", type=str, default="dataset/dataset/crello_images")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--method", type=str, default="ours")
    args = parser.parse_args()
    filename = args.pred
    image_base = args.image_base
    output_dir = args.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "ours":
        if filename.endswith(".json"):
            pred = read_json(filename)
        elif filename.endswith(".jsonl"):
            pred = read_jsonl(filename)

    for sample in tqdm(pred):
        idx = sample["id"]
        num = sample["num"]
        render_image = sample["render_image"]
        render_text = sample["render_text"]
        prediction = sample["predictions"][0]
        canvas_width = sample["canvas_width"]
        canvas_height = sample["canvas_height"]
        try:
            image = render(prediction, image_base, render_image, render_text, canvas_width, canvas_height)
        except:
            image = Image.new("RGB", (canvas_width, canvas_height), "white")
            print(num, idx)
        image.save(output_dir / f"{num}_{idx}.png")


if __name__ == "__main__":
    main()
