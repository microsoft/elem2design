import json
import pickle

import yaml


def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def write_json(filename, data, indent=4):
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent)
    return


def read_pkl(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_pkl(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def read_txt(filename):
    with open(filename, "r") as f:
        return f.read().strip("\n").split("\n")


def read_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
