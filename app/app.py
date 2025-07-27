import gradio as gr
import json
from llava.infer.infer import white_rgb_convert, expand2square
import argparse
from llava.mm_utils import tokenizer_image_token
from llava.model import *
from llava.model.builder import load_pretrained_model
from PIL import Image
import os
import torch
from llava.conversation import layout_conv
from common.context import ContextHandler
from crello.util import render


title = r"""
<h1 align="center">From Elements to Design: A Layered Approach for Automatic Graphic Design Composition</h1>
<h1 align="center">
<a href="https://arxiv.org/abs/2412.19712" style="margin: 30px;">arXiv</a>
<a href="https://github.com/microsoft/elem2design" style="margin: 30px;">GitHub</a>
<a href="https://elements2design.github.io/" style="margin: 30px;">Page</a>
<a href="https://www.youtube.com/watch?v=omXtLEiwEPU" style="margin: 30px;">Video</a>
</h1>
"""

LABEL2INDEX = {"Background": 0, "Underlay": 1, "Image": 2, "Text": 3, "Embellishment": 4}


def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)


def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def on_gallery_click(block_id, select_data: gr.SelectData):
    selected_index = select_data.index
    return block_id, selected_index


def move_btn_click(background_elements, underlay_elements, image_elements, embellishment_elements, selected_block, selected_element, target_block):
    gallery = [background_elements, underlay_elements, image_elements, embellishment_elements]
    if not selected_block in [0, 1, 2, 3]:
        gr.Info("Please choose an element first!")
        return *gallery, gr.State(), gr.State()
    if gallery[target_block]:
        gallery[target_block].append(gallery[selected_block][selected_element])
    else:
        gallery[target_block] = [gallery[selected_block][selected_element]]
    gallery[selected_block].pop(selected_element)
    return *gallery, gr.State(), gr.State()


def on_selection_change(selected_value):
    layers = ["Background", "Underlay", "Image", "Text", "Embellishment"]
    if selected_value == "Background":
        return gr.update(choices=layers[0:])
    if selected_value == "Underlay":
        return gr.update(choices=layers[1:])
    if selected_value == "Image":
        return gr.update(choices=layers[2:])
    if selected_value == "Text":
        return gr.update(choices=layers[3:])
    if selected_value == "Embellishment":
        return gr.update(choices=layers[4:])


def run_stage_1(upload_images):
    gr.Info("please implement the layer planning stage")
    return [], [], [], []


def construct_conversations(
    background_elements, underlay_elements, image_elements, text_elements, embellishment_elements, canvas_width, canvas_height
):
    images = []
    texts = []
    counts = []
    images.extend(background_elements)
    texts.extend([""] * len(background_elements))
    counts.append(len(background_elements))
    images.extend(underlay_elements)
    texts.extend([""] * len(underlay_elements))
    counts.append(len(underlay_elements))
    images.extend(image_elements)
    texts.extend([""] * len(image_elements))
    counts.append(len(image_elements))
    images.extend([""] * len(text_elements))
    texts.extend(text_elements)
    counts.append(len(text_elements))
    images.extend(embellishment_elements)
    texts.extend([""] * len(embellishment_elements))
    counts.append(len(embellishment_elements))
    context_handler = ContextHandler({"template": "index-content"}, images, texts)
    context, _ = context_handler.construct_context()

    start_idx = 0
    context_layer = []
    for cnt in counts:
        context_, _ = context_handler.construct_context(start_index=start_idx, end_index=start_idx + cnt, index_offset=start_idx)
        context_layer.append(context_)
        start_idx += cnt

    return (
        [
            {
                "from": "human",
                "value": f"A poster of canvas width {canvas_width}px, canvas height {canvas_height}px. {context} Please predict step by step according to the semantics of the elements. After each prediction, there will be an intermediate rendering result as a reference to better make the next prediction.\n\n\nNow predict the background elements: {context_layer[0]}",
            },
            {"from": "gpt", "value": ""},
            {"from": "human", "value": f"current canvas state: <image>. Now predict the underlay elements: {context_layer[1]}"},
            {"from": "gpt", "value": ""},
            {"from": "human", "value": f"current canvas state: <image>. Now predict the logo/image elements: {context_layer[2]}"},
            {"from": "gpt", "value": ""},
            {"from": "human", "value": f"current canvas state: <image>. Now predict the text elements: {context_layer[3]}"},
            {"from": "gpt", "value": ""},
            {"from": "human", "value": f"current canvas state: <image>. Now predict the embellishment elements: {context_layer[4]}"},
            {"from": "gpt", "value": ""},
        ],
        images,
        texts,
    )


def get_item(
    background_elements,
    underlay_elements,
    image_elements,
    text_elements,
    embellishment_elements,
    image_list,
    canvas_width,
    canvas_height,
    pred_to=0,
    gpt_dict={},
    images=None,
    new_images={},
):
    layer_image_list = []
    if images is None:
        images = []
        for image_idx, image_file in enumerate(image_list):
            if "layer_" not in image_file:
                try:
                    image = Image.open(image_file)
                    image = white_rgb_convert(image)
                    image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
                    image = image_processor.preprocess(image, return_tensors="pt", input_data_format="channels_last")["pixel_values"][0]
                except:
                    image = torch.zeros(3, 336, 336)
            else:
                layer_image_list.append(image_idx)
                image = torch.zeros(3, 336, 336)
            images.append(image)
    else:
        for k, v in new_images.items():
            image = Image.open(v)
            image = white_rgb_convert(image)
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt", input_data_format="channels_last")["pixel_values"][0]
            images[k] = image

    conversations_, render_image, render_text = construct_conversations(
        background_elements,
        underlay_elements,
        image_elements,
        text_elements,
        embellishment_elements,
        canvas_width,
        canvas_height,
    )

    conv = layout_conv.copy()
    conv.sep2 = tokenizer.eos_token
    conversations = []
    turn_id = 0
    for sentence in conversations_:
        if sentence["from"] == "gpt" and turn_id in gpt_dict:
            sentence["value"] = gpt_dict[turn_id]
        conv.append_message(sentence["from"], sentence["value"])
        if sentence["from"] == "gpt" and turn_id == pred_to:
            break
        if sentence["from"] == "gpt":
            turn_id += 1
    conversations.append(conv.get_prompt())

    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    data_dict = dict(input_ids=input_ids[0])

    if len(images) == 0:
        crop_size = image_processor.crop_size
        data_dict["image"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])
    else:
        data_dict["image"] = torch.stack(images, dim=0)
    data_dict["id"] = ""
    data_dict["render_image"] = render_image
    data_dict["render_text"] = render_text
    data_dict["conversations"] = conversations_
    data_dict["canvas_width"] = canvas_width
    data_dict["canvas_height"] = canvas_height
    return data_dict, images, layer_image_list


def run_stage_2(
    background_elements,
    underlay_elements,
    image_elements,
    text_elements,
    embellishment_elements,
    canvas_width,
    canvas_height,
    temperature,
    top_p,
    pred_from,
    pred_to,
    predictions,
):
    pred_from = LABEL2INDEX[pred_from]
    pred_to = LABEL2INDEX[pred_to]
    for i in range(5):
        if predictions[i] is None and i not in range(pred_from, pred_to + 1):
            gr.Info("please predict from background to embellishment first")
            return []

    with torch.inference_mode():
        background_elements = [ele[0] for ele in background_elements or []]
        underlay_elements = [ele[0] for ele in underlay_elements or []]
        image_elements = [ele[0] for ele in image_elements or []]
        text_elements = text_elements.split("\n")
        text_elements = [ele.replace("\\n", "\n") for ele in text_elements or []]
        embellishment_elements = [ele[0] for ele in embellishment_elements or []]

        image_list = (
            background_elements
            + underlay_elements
            + image_elements
            + embellishment_elements
            + background_elements
            + ["layer_0.png"]
            + underlay_elements
            + ["layer_1.png"]
            + image_elements
            + ["layer_2.png"]
            + ["layer_3.png"]
            + embellishment_elements
        )

        gpt_dict = dict()
        res = []
        sample, processed_images, layer_image_list = get_item(
            background_elements,
            underlay_elements,
            image_elements,
            text_elements,
            embellishment_elements,
            image_list,
            canvas_width,
            canvas_height,
        )
        new_images = {}
        data = {
            "num": "",
            "id": sample["id"],
            "render_image": sample["render_image"],
            "render_text": sample["render_text"],
            "predictions": [""],
            "canvas_width": sample["canvas_width"],
            "canvas_height": sample["canvas_height"],
        }
        for turn_id in range(5):
            if turn_id not in range(pred_from, pred_to + 1):
                data["predictions"] = [item + " ##### " + predictions[turn_id] + " $$$$$ " for item in data["predictions"]]
                gpt_dict[turn_id] = predictions[turn_id]
            else:
                gpt_dict[turn_id] = None
                sample, processed_images, _ = get_item(
                    background_elements,
                    underlay_elements,
                    image_elements,
                    text_elements,
                    embellishment_elements,
                    image_list,
                    canvas_width,
                    canvas_height,
                    pred_to=turn_id,
                    gpt_dict=gpt_dict,
                    images=processed_images,
                    new_images=new_images,
                )
                input_ids = sample["input_ids"].unsqueeze(0).to("cuda")
                images = [sample["image"].to("cuda", dtype=torch.float16)]
                attention_mask = input_ids.ne(tokenizer.pad_token_id).to("cuda")

                try:
                    output_ids = model.generate(
                        input_ids,
                        images=images,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id,
                        max_length=5000,
                    )
                    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                    print(output)
                    data["predictions"] = [item + " ##### " + output + " $$$$$ " for item in data["predictions"]]
                    predictions[turn_id] = output
                    gpt_dict[turn_id] = output

                except Exception as e:
                    gpt_dict[turn_id] = "{}"

            # render
            try:
                render_image = render(
                    data["predictions"][0], "/", data["render_image"], data["render_text"], data["canvas_width"], data["canvas_height"]
                )
            except:
                render_image = Image.new("RGB", (data["canvas_width"], data["canvas_height"]), color="white")
            render_image_filename = os.path.join("./app", f"{turn_id}.png")
            render_image.save(render_image_filename)
            res.append(render_image_filename)
            if turn_id < 4:
                new_images = {layer_image_list[turn_id]: render_image_filename}

    return res


with gr.Blocks() as demo:
    gr.Markdown(title)
    selected_block = gr.State()
    selected_element = gr.State()
    predictions = gr.State([None, None, None, None, None])

    with gr.Row():
        with gr.Column():
            gr.Markdown("""<h2 align="left">1️⃣ Layer division</h2>""")

            with gr.Accordion("Layer Planning (Optional)", open=False):
                files = gr.File(file_count="multiple", file_types=["image"], label="Drag (Select) design elements")
                uploaded_files = gr.Gallery(label="Uploaded elements", visible=False, columns=5, object_fit="contain", height="150px")
                with gr.Column(visible=False) as clear_button:
                    remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")
                layer_planning_button = gr.Button(value="Submit (Stage I)", visible=False)

            with gr.Row(equal_height=True):
                background_elements = gr.Gallery(label="Background", object_fit="contain", height="200px")
                underlay_elements = gr.Gallery(label="Underlay", object_fit="contain", height="200px")

            with gr.Row():
                background_btn = gr.Button(value="Move to", size="sm")
                underlay_btn = gr.Button(value="Move to", size="sm")

            with gr.Row(equal_height=True):
                image_elements = gr.Gallery(label="Image", object_fit="contain", height="200px")
                embellishment_elements = gr.Gallery(label="Embellishment", object_fit="contain", height="200px")

            with gr.Row():
                image_btn = gr.Button(value="Move to", size="sm")
                embellishment_btn = gr.Button(value="Move to", size="sm")

            text_elements = gr.Textbox(lines=2, label="Text", info="enter text content here, each text element occupies one line")

        with gr.Column():
            gr.Markdown("""<h2 align="left">2️⃣ Compose elements into design</h2>""")
            output_gallery = gr.Gallery(label="Generated Design", columns=5, object_fit="contain")
            with gr.Row():
                canvas_width = gr.Number(label="Canvas Width (px)")
                canvas_height = gr.Number(label="Canvas Height (px)")

            with gr.Row():
                pred_from = gr.Dropdown(label="Predict from", choices=["Background", "Underlay", "Image", "Text", "Embellishment"], interactive=True)
                pred_to = gr.Dropdown(
                    label="to", interactive=True, value="Embellishment", choices=["Background", "Underlay", "Image", "Text", "Embellishment"]
                )
            with gr.Accordion(open=False, label="Advanced Options"):
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.7, interactive=True)
                top_p = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=0.95, interactive=True)
            compose_button = gr.Button(value="Submit (Stage II)")

    files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, layer_planning_button, files])
    remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, layer_planning_button, files])
    background_elements.select(fn=on_gallery_click, inputs=gr.Number(0, visible=False), outputs=[selected_block, selected_element])
    underlay_elements.select(fn=on_gallery_click, inputs=gr.Number(1, visible=False), outputs=[selected_block, selected_element])
    image_elements.select(fn=on_gallery_click, inputs=gr.Number(2, visible=False), outputs=[selected_block, selected_element])
    embellishment_elements.select(fn=on_gallery_click, inputs=gr.Number(3, visible=False), outputs=[selected_block, selected_element])

    move_list = [
        background_elements,
        underlay_elements,
        image_elements,
        embellishment_elements,
        selected_block,
        selected_element,
    ]

    background_btn.click(fn=move_btn_click, inputs=move_list + [gr.Number(0, visible=False)], outputs=move_list)
    underlay_btn.click(fn=move_btn_click, inputs=move_list + [gr.Number(1, visible=False)], outputs=move_list)
    image_btn.click(fn=move_btn_click, inputs=move_list + [gr.Number(2, visible=False)], outputs=move_list)
    embellishment_btn.click(fn=move_btn_click, inputs=move_list + [gr.Number(3, visible=False)], outputs=move_list)
    layer_planning_button.click(
        fn=run_stage_1, inputs=uploaded_files, outputs=[background_elements, underlay_elements, image_elements, embellishment_elements]
    )
    compose_button.click(
        fn=run_stage_2,
        inputs=[
            background_elements,
            underlay_elements,
            image_elements,
            text_elements,
            embellishment_elements,
            canvas_width,
            canvas_height,
            temperature,
            top_p,
            pred_from,
            pred_to,
            predictions,
        ],
        outputs=output_gallery,
    )

    pred_from.change(fn=on_selection_change, inputs=pred_from, outputs=pred_to)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    args = parser.parse_args()
    model_path = args.model_name_or_path
    with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
        model_base = json.load(f)["base_model_name_or_path"]
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base)
    tokenizer.pad_token_id = tokenizer.unk_token_id or 0
    model = model.to("cuda")

    demo.launch(share=True, server_port=5678)
