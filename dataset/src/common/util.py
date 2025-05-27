import math


def postprocess(data, canvas_width: int, canvas_height: int):
    if "left" in data and data["left"] is not None:
        data["left"] = round(data["left"] * canvas_width)
    if "top" in data and data["top"] is not None:
        data["top"] = round(data["top"] * canvas_height)
    if "width" in data and data["width"] is not None:
        data["width"] = round(data["width"] * canvas_width)
    if "height" in data and data["height"] is not None:
        data["height"] = round(data["height"] * canvas_height)
    if "angle" in data and data["angle"] is not None:
        data["angle"] = round(math.degrees(data["angle"])) % 360
    if "font_size" in data and data["font_size"] is not None:
        data["font_size"] = round(data["font_size"])
    if "color" in data and data["color"] is not None:
        data["color"] = [round(data["color"][0]), round(data["color"][1]), round(data["color"][2])]
    if "letter_spacing" in data and data["letter_spacing"] is not None:
        data["letter_spacing"] = round(float(data["letter_spacing"]), 1)
    if "line_height" in data and data["line_height"] is not None:
        data["line_height"] = round(float(data["line_height"]), 1)
    return data


def remove_unpredicted_keys(data):
    remove_keys = ["angle", "font", "font_size", "color", "text_align", "capitalize", "letter_spacing", "line_height"]
    if data["text"] == "":
        for key in remove_keys:
            data.pop(key)
    data.pop("text")
    return data


def add_column(dataset, col_dict: dict, key: str = "id", col_name: str = "new_column", default_value=None):
    def _add_column(example):
        idx = example[key]
        if idx in col_dict:
            value = col_dict[idx]
        else:
            value = default_value
        example[col_name] = value
        return example

    dataset = dataset.map(_add_column)
    return dataset
