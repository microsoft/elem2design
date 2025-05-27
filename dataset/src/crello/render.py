import io
from functools import partial
from pathlib import Path

from common.util import add_column
from crello.util import role_factory
from layoutlib.hfds import hfds_factory, hfds_helper_factory
from layoutlib.hfds.util import filter_layer
from opencole.renderer.renderer import ExampleRenderer
from PIL import Image
from tqdm import tqdm

DATASET = "crello"


def role_filter(example, index, role):
    return example["role"][index] <= role


role_filter0 = partial(role_filter, role=0)
role_filter1 = partial(role_filter, role=1)
role_filter2 = partial(role_filter, role=2)
role_filter3 = partial(role_filter, role=3)
role_filter4 = partial(role_filter, role=4)


def main() -> None:
    output_dir = Path("dataset/dataset/crello_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = hfds_factory(DATASET)
    features = dataset_dict["train"].features
    hfds_helper = hfds_helper_factory(DATASET, features=features)
    renderer = ExampleRenderer(features=hfds_helper.renderer_features)
    role_dict = role_factory()

    for split in dataset_dict:
        dataset = dataset_dict[split]
        dataset = add_column(dataset, col_dict=role_dict, col_name="role")

        for i, example in enumerate(tqdm(dataset)):
            id_ = example["id"]
            (output_dir / id_).mkdir(parents=True, exist_ok=True)
            canvas_width, canvas_height = hfds_helper.get_canvas_size(example)
            example_layer_0 = filter_layer(example, role_filter0)
            example_layer_1 = filter_layer(example, role_filter1)
            example_layer_2 = filter_layer(example, role_filter2)
            example_layer_3 = filter_layer(example, role_filter3)
            example_layer_4 = filter_layer(example, role_filter4)

            # render
            image_bytes_0 = renderer.render(
                example=hfds_helper.convert_for_renderer(example_layer_0),
                max_size=max(canvas_width, canvas_height),
            )
            image_bytes_1 = renderer.render(
                example=hfds_helper.convert_for_renderer(example_layer_1),
                max_size=max(canvas_width, canvas_height),
            )
            image_bytes_2 = renderer.render(
                example=hfds_helper.convert_for_renderer(example_layer_2),
                max_size=max(canvas_width, canvas_height),
            )
            image_bytes_3 = renderer.render(
                example=hfds_helper.convert_for_renderer(example_layer_3),
                max_size=max(canvas_width, canvas_height),
            )
            image_bytes_4 = renderer.render(
                example=hfds_helper.convert_for_renderer(example_layer_4),
                max_size=max(canvas_width, canvas_height),
            )

            image0 = Image.open(io.BytesIO(image_bytes_0))
            image1 = Image.open(io.BytesIO(image_bytes_1))
            image2 = Image.open(io.BytesIO(image_bytes_2))
            image3 = Image.open(io.BytesIO(image_bytes_3))
            image4 = Image.open(io.BytesIO(image_bytes_4))

            image0.save(output_dir / id_ / f"layer_0.png")
            image1.save(output_dir / id_ / f"layer_1.png")
            image2.save(output_dir / id_ / f"layer_2.png")
            image3.save(output_dir / id_ / f"layer_3.png")
            image4.save(output_dir / id_ / f"layer_4.png")

            for num, (ele, ele_w, ele_h) in enumerate(zip(example["image"], example["width"], example["height"])):
                ele_w = max(1, int(ele_w * canvas_width))
                ele_h = max(1, int(ele_h * canvas_height))
                ele = ele.resize((ele_w, ele_h))
                ele.save(output_dir / id_ / f"ele_{num}.png")


if __name__ == "__main__":
    main()
