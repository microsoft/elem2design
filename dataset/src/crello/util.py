import io
import os
import re
from copy import deepcopy
from pathlib import Path

import datasets as ds
from common.io import read_json, read_pkl, write_pkl
from langchain.output_parsers import PydanticOutputParser
from layoutlib.hfds import sample_example
from layoutlib.hfds.crello import ADDITIONAL_FONT_PROPERTIES
from layoutlib.hfds.util import fill_missing_values
from layoutlib.schema import element_schema_factory, get_element_pydantic_model
from layoutlib.util import list_of_dict_to_dict_of_list
from opencole.renderer.auto_line import wrap_text_to_fit_box
from opencole.renderer.renderer import ExampleRenderer
from PIL import Image
from tqdm import tqdm


ROLE2INDEX = {"Background": 0, "Underlay": 1, "Logo/Image": 2, "Logo": 2, "Image": 2, "Text": 3, "Embellishment": 4}


INDEX2ROLE = {0: "Background", 1: "Underlay", 2: "Logo/Image", 3: "Text", 4: "Embellishment"}


CRELLO_ROLE_PATH = Path(__file__).parent.parent.parent / "dataset" / "role" / "crello_role.pkl"


def role_factory(folder: str = None, overwrite: bool = False):
    role_path = CRELLO_ROLE_PATH
    if role_path.exists() and not overwrite:
        role = read_pkl(role_path)
    else:
        assert folder is not None
        folder = Path(folder)
        role = {}
        for sub_folder in tqdm(folder.glob("*"), desc="process role..."):
            idx = sub_folder.stem
            role[idx] = list()
            for ele in sorted(sub_folder.glob("*.json"), key=lambda x: int(x.stem)):
                ele = read_json(ele)
                role[idx].append(ROLE2INDEX.get(ele["role"], 4))
        write_pkl(role_path, role)
    return role


def ele_image_factory(folder: str):
    folder = Path(folder)
    image_path = {}
    for sub_folder in tqdm(folder.glob("*"), desc="processing ele image..."):
        idx = sub_folder.stem
        image_path[idx] = sorted(sub_folder.glob("ele_*.png"), key=lambda x: int(x.stem.replace("ele_", "")))
        image_path[idx] = [str(Path(*path.parts[-2:])) for path in image_path[idx]]
    return image_path


CRELLO_FEATURES = ds.Features(
    {
        "id": ds.Value("string"),
        "length": ds.Value("int32"),
        "canvas_width": ds.Value("int32"),
        "canvas_height": ds.Value("int32"),
        "preview": ds.Image(),
        "type": ds.Sequence(ds.Value("string")),
        "left": ds.Sequence(ds.Value("float32")),
        "top": ds.Sequence(ds.Value("float32")),
        "width": ds.Sequence(ds.Value("float32")),
        "height": ds.Sequence(ds.Value("float32")),
        "angle": ds.Sequence(ds.Value("float32")),
        "image": ds.Sequence(ds.Image()),
        "text": ds.Sequence(ds.Value("string")),
        "font": ds.Sequence(ds.Value("string")),
        "font_size": ds.Sequence(ds.Value("float32")),
        "font_bold": ds.Sequence(ds.Sequence(ds.Value("bool"))),
        "font_italic": ds.Sequence(ds.Sequence(ds.Value("bool"))),
        "text_color": ds.Sequence(ds.Sequence(ds.Value("string"))),
        "text_line": ds.Sequence(ds.Sequence(ds.Value("int32"))),
        "text_align": ds.Sequence(ds.Value("string")),
        "capitalize": ds.Sequence(ds.Value("bool")),
        "line_height": ds.Sequence(ds.Value("float32")),
        "letter_spacing": ds.Sequence(ds.Value("float32")),
    }
)


def convert_for_renderer(example: dict):
    example_new = deepcopy(example)
    N = example_new["length"]

    # set initial values
    example_new["text_color"] = [[] for _ in range(N)]
    example_new["text_line"] = [[] for _ in range(N)]
    example_new["type"] = [None for _ in range(N)]
    for key in ADDITIONAL_FONT_PROPERTIES:
        example_new[f"font_{key}"] = [[] for _ in range(N)]

    for i in range(example_new["length"]):
        color = example_new["color"][i]
        example_new["angle"][i] = example_new["angle"][i] or 0.0
        if example_new["text"][i] != "":
            text = example_new["text"][i]
            example_new["type"][i] = "textElement"

            text_line, line_index = [], 0
            for char in text:
                if char == "\n":
                    line_index += 1
                text_line.append(line_index)
            example_new["text_line"][i] = text_line

            try:
                example_new["text_color"][i] = [f"rgba({int(color[0])}, {int(color[1])}, {int(color[2])}, 1)" for _ in range(len(text))]
            except:
                example_new["text_color"][i] = ["rgba(255, 255, 255, 1)" for _ in range(len(text))]

            for key, prop in ADDITIONAL_FONT_PROPERTIES.items():
                example_new[f"font_{key}"][i] = [prop["default"] for _ in range(len(text))]
        else:
            example_new["type"][i] = "imageElement"

    example_new["capitalize"] = [num == "true" for num in example_new["capitalize"]]

    del example_new["color"]

    return example_new


_, features = sample_example("crello")
renderer = ExampleRenderer(features=CRELLO_FEATURES)
element_parser = PydanticOutputParser(  # type: ignore
    pydantic_object=get_element_pydantic_model(
        schema=element_schema_factory("vista"),
    )
)


def render(prediction, image_base, render_image, render_text, canvas_width, canvas_height, text_from="GT", auto_newline=False):
    layout = []
    elements = re.findall(r"{.*?}", prediction, re.DOTALL)
        
    for element in elements:
        if element != "{}":
            try:
                element = element_parser.parse(element)
                element = element.dict()
                if text_from == "GT":
                    element["text"] = render_text[element["index"]]
                elif text_from == "pred":
                    assert "text" in element and element["text"] is not None

                if auto_newline:
                    element["text"] = element["text"].replace("\n", " ")
                    element["text"] = wrap_text_to_fit_box(
                        element["width"], element["text"], element["font"], element["font_size"], element["letter_spacing"], element["capitalize"]
                    )
                try:
                    element["image"] = Image.open(os.path.join(image_base, render_image[element["index"]]))
                except:
                    element["image"] = Image.open("app/placeholder.png")
                layout.append(element)
            except Exception as e:
                print(e)
                pass

    hfds = list_of_dict_to_dict_of_list(layout)
    hfds["length"] = len(hfds["index"])
    hfds["canvas_width"] = canvas_width
    hfds["canvas_height"] = canvas_height
    image_bytes = renderer.render(example=fill_missing_values(convert_for_renderer(hfds), features), max_size=max(canvas_width, canvas_height))
    image = Image.open(io.BytesIO(image_bytes))
    return image
