# From Elements to Design: A Layered Approach for Automatic Graphic Design Composition

[[Paper](https://arxiv.org/abs/2412.19712)]
[[Video](https://www.youtube.com/watch?v=omXtLEiwEPU)]
[[Hugging Face](https://huggingface.co/microsoft/elem2design)]
[[Demo](app/app.py)]

In this work, we investigate automatic design composition from multimodal graphic elements.
We propose [LaDeCo](https://arxiv.org/pdf/2412.19712), which introduces the layered design principle to accomplish this challenging task through two steps: layer planning and layered design composition.

![](teaser.png)

# Requirements

1. Clone this repository
```bash
git clone https://github.com/microsoft/elem2design.git
cd elem2design
```

2. Install
```bash
conda create -n e2d python=3.10 -y
conda activate e2d
pip install --upgrade pip
pip install -e .
pip install -e thirdparty/opencole
pip install -e dataset/src
```

3. Install additional packages for training cases
```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

# How to Use

## Layer Planning

Please check this [folder](dataset/labeling) for the `layer planning` code, or you can directly use the predicted labels [here](dataset/dataset/role/crello_role.pkl).

Index-to-layer mapping:
```python
{
    0: "Background", 
    1: "Underlay", 
    2: "Logo/Image", 
    3: "Text", 
    4: "Embellishment"
}
```

## Rendering

We render an image for each element, which is useful during inference.
Meanwhile, we also render the intermediate designs (denoted as `layer_{index}.png`) and use them for end-to-end training.

Please run the following script to get these assets:
```bash
python dataset/src/crello/render.py
```

## Dataset Preparation

After rendering, we move to the next step to construct the dataset according to layered design principle. 
Each sample has 5 rounds of dialogue, where the model progressively predicts element attributes from the background layer to the embellishment layer.

```bash
python dataset/src/crello/create_dataset.py --tag ours
```

## Model

We have a model available on [Hugging Face](https://huggingface.co/microsoft/elem2design).
Please download it for inference.

This model is trained on the public [Crello dataset](https://huggingface.co/datasets/cyberagent/crello).
We find that using a dataset approximately five times larger leads to significantly improved performance. For a detailed evaluation, please refer to Table 2 in our [paper](https://arxiv.org/pdf/2412.19712). Unfortunately, we are unable to release this model as it was trained on a private dataset.

## Inference

Now it is time to do inference using the prepared data and model:
```bash
python llava/infer/infer.py \
    --model_name_or_path /path/to/model/checkpoint-xxxx \
    --data_path /path/to/data/test.json \
    --image_folder /path/to/crello_images \
    --output_dir /path/to/output_dir \
    --start_layer_index 0 \ 
    --end_layer_index 4
```

# Demo

Besides command-line inference, we also provide a demo interface that allows users to interact with the model via a web-based UI. This interface makes it more user-friendly and better suited for running inference on custom datasets.

To launch the web UI, run the following command:

```bash
python app/app.py --model_name_or_path /path/to/model/checkpoint-xxxx
```

# Evaluation

We compute the LVM scores and geometry-related metrics for the generated designs:

- [LVM scores](llava/metrics/llava_ov.py)
```bash
python llava/metrics/llava_ov.py -i /path/to/output_dir
```

- [Geometry-related metrics](llava/metrics/layout.py)
```bash
python llava/metrics/layout.py --pred /path/to/output_dir/pred.jsonl
```

# Training

We fine-tune LLMs using the `crello` training set for layered design composition.
For your own dataset, please prepare the dataset in the given format and run:
```bash
bash scripts/finetune_lora.sh \
    1 \
    meta-llama/Llama-3.1-8B \
    /path/to/dataset/train.json \
    /path/to/image/base \
    /path/to/output_dir \
    50 \
    2 \
    16 \
    250 \
    2e-4 \
    2e-4 \
    cls_pooling \
    Llama-3.1-8B_lora_ours \
    32 \
    64 \
    4
```

For example, the specific script in our setting is:
```bash
bash scripts/finetune_lora.sh \
    1 \
    meta-llama/Llama-3.1-8B \
    dataset/dataset/json/ours/train.json \
    dataset/dataset/crello_images \
    output/Llama-3.1-8B_lora_ours \
    50 \
    2 \
    16 \
    250 \
    2e-4 \
    2e-4 \
    cls_pooling \
    Llama-3.1-8B_lora_ours \
    32 \
    64 \
    4
```

Remember to login to [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B) using the Llama access token:
```bash
huggingface-cli login --token $TOKEN
```

The following is a list of supported LLMs:

- [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b)
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)

# BibTeX

```
@InProceedings{lin2024elements,
  title={From Elements to Design: A Layered Approach for Automatic Graphic Design Composition},
  author={Lin, Jiawei and Sun, Shizhao and Huang, Danqing and Liu, Ting and Li, Ji and Bian, Jiang},
  booktitle={CVPR},
  year={2025}
}
```

## Acknowledgments

We would like to express our gratitude to [CanvasVAE](https://huggingface.co/datasets/cyberagent/crello) for providing the dataset, [OpenCole](https://github.com/CyberAgentAILab/OpenCOLE) for the rendering code, and [LLaVA](https://github.com/haotian-liu/LLaVA) for the codebase. 
We deeply appreciate all the incredible work that made this project possible.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
