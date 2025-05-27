import argparse
import json
from pathlib import Path

from common.context import ContextHandler
from common.io import read_yaml, write_json
from common.util import add_column, postprocess, remove_unpredicted_keys
from crello.util import ele_image_factory, role_factory
from layoutlib.hfds import hfds_factory, hfds_helper_factory
from layoutlib.hfds.util import extract_class_label_mappings, shuffle_transform, sort_transform
from layoutlib.manager import LayoutManager
from opencole.preprocess.language import is_standard
from opencole.util import set_seed
from tqdm import tqdm

DATASET = "crello"


def main() -> None:
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--output_dir", type=str, default="./dataset/dataset/json")
    parser.add_argument("--tag", type=str, default=None, help="dataset tag")
    parser.add_argument("--config", type=str, default="dataset/src/crello/config/crello_v1.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--raw_role_folder", type=str, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    seed = config.get("seed", 0)
    max_num = config.get("max_num", -1)
    schema_name = config["manager"]["schema_name"]
    set_seed(seed)

    output_dir = Path(args.output_dir)
    if args.tag is not None:
        output_dir = output_dir / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = hfds_factory(DATASET)
    features = dataset_dict["train"].features
    hfds_helper = hfds_helper_factory(DATASET, features=features)
    class_label_mappings = extract_class_label_mappings(features)
    manager = LayoutManager(
        **config["manager"],
        class_label_mappings=class_label_mappings,
    )

    role_dict = role_factory(folder=args.raw_role_folder)
    image_path_dict = ele_image_factory(folder="./dataset/dataset/crello_images")

    for split in dataset_dict:
        annotations = []
        dataset = dataset_dict[split]
        dataset = add_column(dataset, col_dict=image_path_dict, col_name="image_path")
        dataset = add_column(dataset, col_dict=role_dict, col_name="role")

        for i, example in enumerate(tqdm(dataset)):
            id_ = example["id"]
            canvas_width, canvas_height = hfds_helper.get_canvas_size(example)

            # set filtering rules here
            if not all([is_standard(t) for t in example["text"]]) and split in ["train", "validation"]:
                continue

            if max_num > 0 and example["length"] > max_num and split in ["train", "validation"]:
                continue

            # shuffle, prevent information leakage
            example = shuffle_transform(example)
            example = sort_transform(example, key="role")

            example["index"] = list(range(example["length"]))
            context_handler = ContextHandler(config=config["context"], images=example["image_path"], texts=example["text"])
            context, image_path = context_handler.construct_context()

            if schema_name in ["kmeans", "linear"]:
                example = hfds_helper.normalize(example)
                canvas_width, canvas_height = 128, 128
            layout = manager.hfds_to_layout_instance(example)

            num_ele_per_layer = {layer: example["role"].count(layer) for layer in range(5)}
            context_by_layer = []
            layout_seq_by_layer = []
            cnt = 0
            for layer, num_ele in num_ele_per_layer.items():
                _context, _image_path = context_handler.construct_context(start_index=cnt, end_index=cnt + num_ele, index_offset=cnt)
                context_by_layer.append(_context)
                image_path.extend(_image_path)
                if layer < 4:
                    image_path.append(f"{id_}/layer_{layer}.png")

                layout_seq = []
                for element in layout.elements[cnt : cnt + num_ele]:
                    element_seq = element.dict()
                    if schema_name == "default":
                        element_seq = postprocess(element_seq, canvas_width=canvas_width, canvas_height=canvas_height)
                    element_seq = remove_unpredicted_keys(element_seq)
                    layout_seq.append(json.dumps(element_seq))
                layout_seq_by_layer.append(layout_seq)
                cnt += num_ele

            preamble = "A poster of canvas width {}px, canvas height {}px. ".format(canvas_width, canvas_height)
            annotation = {
                "id": id_,
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": preamble + context + " Please predict step by step according to the semantics of the elements. After each prediction, there will be an intermediate rendering result as a reference to better make the next prediction.\n\n\n" f"Now predict the background elements: {context_by_layer[0]}",
                    },
                    {"from": "gpt", "value": " ".join(layout_seq_by_layer[0]) if len(layout_seq_by_layer[0]) > 0 else "{}"},
                    {"from": "human", "value": f"current canvas state: <image>. Now predict the underlay elements: {context_by_layer[1]}"},
                    {"from": "gpt", "value": " ".join(layout_seq_by_layer[1]) if len(layout_seq_by_layer[1]) > 0 else "{}"},
                    {"from": "human", "value": f"current canvas state: <image>. Now predict the logo/image elements: {context_by_layer[2]}"},
                    {"from": "gpt", "value": " ".join(layout_seq_by_layer[2]) if len(layout_seq_by_layer[2]) > 0 else "{}"},
                    {"from": "human", "value": f"current canvas state: <image>. Now predict the text elements: {context_by_layer[3]}"},
                    {"from": "gpt", "value": " ".join(layout_seq_by_layer[3]) if len(layout_seq_by_layer[3]) > 0 else "{}"},
                    {"from": "human", "value": f"current canvas state: <image>. Now predict the embellishment elements: {context_by_layer[4]}"},
                    {"from": "gpt", "value": " ".join(layout_seq_by_layer[4]) if len(layout_seq_by_layer[4]) > 0 else "{}"},
                ],
                "render_image": example["image_path"],
                "render_text": example["text"],
            }
            annotations.append(annotation)

            if args.debug and i == 0:
                break

        write_json(output_dir / f"{split}.json", annotations)
    write_json(output_dir / "preprocess_args.json", vars(args))


if __name__ == "__main__":
    main()
