# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 20:18:00 2022

@author: shyoh
"""

import torch
import os
import copy
import numpy as np
import cv2
from PIL import Image
from math import log
from common.io import read_json, read_jsonl


gpu = torch.cuda.is_available()
device_ids = [0, 1, 2, 3]
device = torch.device(f"cuda:{device_ids[0]}" if gpu else "cpu")


def cvt_pilcv(img, req="pil2cv", color_code=None):
    if req == "pil2cv":
        if color_code == None:
            color_code = cv2.COLOR_RGB2BGR
        dst = cv2.cvtColor(np.asarray(img), color_code)
    elif req == "cv2pil":
        if color_code == None:
            color_code = cv2.COLOR_BGR2RGB
        dst = Image.fromarray(cv2.cvtColor(img, color_code))
    return dst


def img_to_g_xy(img):
    img_cv_gs = np.uint8(cvt_pilcv(img, "pil2cv", cv2.COLOR_RGB2GRAY))
    # Sobel(src, ddepth, dx, dy)
    grad_x = cv2.Sobel(img_cv_gs, -1, 1, 0)
    grad_y = cv2.Sobel(img_cv_gs, -1, 0, 1)
    grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
    grad_xy = grad_xy / np.max(grad_xy) * 255
    img_g_xy = Image.fromarray(grad_xy).convert("L")
    return img_g_xy


def metrics_iou(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2

    w_1 = xr_1 - xl_1
    w_2 = xr_2 - xl_2
    h_1 = yr_1 - yl_1
    h_2 = yr_2 - yl_2

    w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
    h_inter = min(yr_1, yr_2) - max(yl_1, yl_2)

    a_1 = w_1 * h_1
    a_2 = w_2 * h_2
    a_inter = w_inter * h_inter
    if w_inter <= 0 or h_inter <= 0:
        a_inter = 0

    return a_inter / (a_1 + a_2 - a_inter + 1e-10)


def metrics_inter_oneside(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2

    w_1 = xr_1 - xl_1
    w_2 = xr_2 - xl_2
    h_1 = yr_1 - yl_1
    h_2 = yr_2 - yl_2

    w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
    h_inter = min(yr_1, yr_2) - max(yl_1, yl_2)

    a_1 = w_1 * h_1
    a_2 = w_2 * h_2
    a_inter = w_inter * h_inter
    if w_inter <= 0 or h_inter <= 0:
        a_inter = 0

    return a_inter / a_2


def metrics_val(img_size, clses, boxes, canvas_widths, canvas_heights):
    """
    The ratio of non-empty layouts.
    Higher is better.
    """
    w, h = img_size

    total_elem = 0
    empty_elem = 0

    for cls, box, canvas_width, canvas_height in zip(clses, boxes, canvas_widths, canvas_heights):
        cls = np.array(cls, dtype=int)[:, np.newaxis]
        box = np.array(box, dtype=int)
        mask = (cls > 0).reshape(-1)

        mask_box = box[mask]

        total_elem += len(mask_box)
        for mb in mask_box:
            xl, yl, xr, yr = mb
            xl = max(0, xl)
            yl = max(0, yl)
            xr = min(canvas_width, xr)
            yr = min(canvas_height, yr)
            if abs((xr - xl) * (yr - yl)) < canvas_width * canvas_height / 100 / 100 * 10:
                empty_elem += 1
    return 1 - empty_elem / total_elem


def getRidOfInvalid(img_size, clses, boxes, canvas_widths, canvas_heights):
    w, h = img_size

    for i, (cls, box, canvas_width, canvas_height) in enumerate(zip(clses, boxes, canvas_widths, canvas_heights)):
        for j, b in enumerate(box):
            xl, yl, xr, yr = b
            xl = max(0, xl)
            yl = max(0, yl)
            xr = min(canvas_width, xr)
            yr = min(canvas_height, yr)
            # must greater than 0.1% of canvas
            if abs((xr - xl) * (yr - yl)) < canvas_width * canvas_height / 100 / 100 * 10:
                if clses[i][j]:
                    clses[i][j] = 0
                # if clses[i, j]:
                #    clses[i, j] = 0
    return clses


def metrics_uti(img_names, clses, boxes):
    metrics = 0
    for idx, name in enumerate(img_names):
        pic_1 = np.array(Image.open(os.path.join("dataset/baseline/saliency_isnet", name + ".png")).convert("L")) / 255
        pic_2 = np.array(Image.open(os.path.join("dataset/baseline/saliency_basnet", name + ".png")).convert("L")) / 255

        pic = np.maximum(pic_1, pic_2)
        # pic = np.array(Image.open(os.path.join("data/cgl_dataset/salient_imgs_cgl",
        #                                       name)).convert("L").resize((513, 750))) / 255
        c_pic = np.ones_like(pic) - pic

        cal_mask = np.zeros_like(pic)

        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)

        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1

        total_not_sal = np.sum(c_pic)
        total_utils = np.sum(c_pic * cal_mask)

        if total_not_sal and total_utils:
            metrics += total_utils / total_not_sal
    return metrics / len(img_names)


def metrics_rea(img_names, clses, boxes):
    """
    Average gradients of the pixels covered by predicted text-only elements.
    Lower is better.
    """
    metrics = 0
    for idx, name in enumerate(img_names):
        pic = Image.open(os.path.join("dataset/baseline/PosterLlama/image", name + ".png")).convert("RGB")
        img_g_xy = np.array(img_to_g_xy(pic)) / 255  # gradient
        cal_mask = np.zeros_like(img_g_xy)

        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)

        text = (cls == 4).reshape(-1)
        text_box = box[text]
        deco = (cls == 2).reshape(-1)
        deco_box = box[deco]

        for mb in text_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        for mb in deco_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 0

        total_area = np.sum(cal_mask)  # text box area
        total_grad = np.sum(img_g_xy[cal_mask == 1])  #
        if total_grad and total_area:
            metrics += total_grad / total_area
    return metrics / len(img_names)


def metrics_ove(clses, boxes):
    """
    Ratio of overlapping area.
    Lower is better.
    """
    metrics = 0
    for cls, box in zip(clses, boxes):
        ove = 0
        cls = np.array(cls, dtype=int)[:, np.newaxis]
        box = np.array(box, dtype=int)
        mask = (cls > 0).reshape(-1) & (cls != 2).reshape(-1)
        mask_box = box[mask]
        n = len(mask_box)
        for i in range(n):
            bb1 = mask_box[i]
            for j in range(i + 1, n):
                bb2 = mask_box[j]
                ove += metrics_iou(bb1, bb2)
        try:
            metrics += ove / n
        except:
            pass
    return metrics / len(clses)


def metrics_und_l(clses, boxes):
    """
    Overlap ratio of an underlay(deco) and a max-overlapped non-underlay(deco) element.
    Higher is better.
    """
    metrics = 0
    avali = 0
    for cls, box in zip(clses, boxes):
        und = 0
        cls = np.array(cls, dtype=int)[:, np.newaxis]
        box = np.array(box, dtype=int)
        mask_deco = (cls == 2).reshape(-1)
        mask_other = (cls > 0).reshape(-1) & (cls != 2).reshape(-1)
        box_deco = box[mask_deco]
        box_other = box[mask_other]
        n1 = len(box_deco)
        n2 = len(box_other)
        if n1:
            avali += 1
            for i in range(n1):
                max_ios = 0
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    ios = metrics_inter_oneside(bb1, bb2)
                    max_ios = max(max_ios, ios)
                und += max_ios
            metrics += und / n1
    if avali > 0:
        return metrics / avali
    return 0


def is_contain(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2

    c1 = xl_1 <= xl_2
    c2 = yl_1 <= yl_2
    c3 = xr_2 >= xr_2
    c4 = yr_1 >= yr_2

    return c1 and c2 and c3 and c4


def metrics_und_s(clses, boxes):
    """
    Overlap ratio of an underlay(deco) and a max-overlapped non-underlay(deco) element.
    Higher is better.
    """
    metrics = 0
    avali = 0
    for cls, box in zip(clses, boxes):
        und = 0
        cls = np.array(cls, dtype=int)[:, np.newaxis]
        box = np.array(box, dtype=int)
        mask_deco = (cls == 2).reshape(-1)
        mask_other = (cls > 0).reshape(-1) & (cls != 2).reshape(-1)
        box_deco = box[mask_deco]
        box_other = box[mask_other]
        n1 = len(box_deco)
        n2 = len(box_other)
        if n1:
            avali += 1
            for i in range(n1):
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    if is_contain(bb1, bb2):
                        und += 1
                        break
            metrics += und / n1
    if avali > 0:
        return metrics / avali
    return 0


def ali_g(x):
    if x > 1:
        x = 0.99
    return -log(1 - x, 10)


def ali_delta(xs):
    n = len(xs)
    min_delta = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(xs[i] - xs[j])
            min_delta = min(min_delta, delta)
    return min_delta


def metrics_ali(clses, boxes, canvas_widths, canvas_heights):
    """
    Indicator of the extent of non-alignment of pairs of elements.
    Lower is better.
    """
    metrics = 0
    for cls, box, canvas_width, canvas_height in zip(clses, boxes, canvas_widths, canvas_heights):
        ali = 0
        cls = np.array(cls, dtype=float)[:, np.newaxis]
        box = np.array(box, dtype=float)
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        theda = []
        for mb in mask_box:
            pos = copy.deepcopy(mb)
            pos[0] = max(0, pos[0])
            pos[1] = max(0, pos[1])
            pos[2] = min(canvas_width, pos[2])
            pos[3] = min(canvas_height, pos[3])
            pos[0] /= canvas_width
            pos[2] /= canvas_width
            pos[1] /= canvas_height
            pos[3] /= canvas_height
            theda.append([pos[0], pos[1], (pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2, pos[2], pos[3]])
        theda = np.array(theda)
        if theda.shape[0] <= 1:
            continue

        n = len(mask_box)
        for i in range(n):
            g_val = []
            for j in range(6):
                xys = theda[:, j]
                delta = ali_delta(xys)
                g_val.append(ali_g(delta))
            ali += min(g_val)
        metrics += ali

    return metrics / len(clses)


def metrics_occ(img_names, clses, boxes):
    """
    Average saliency of the pixels covered.
    Lower is better.
    """
    metrics = 0
    for idx, name in enumerate(img_names):
        pic_1 = np.array(Image.open(os.path.join("dataset/baseline/saliency_isnet", name + ".png")).convert("L")) / 255
        pic_2 = np.array(Image.open(os.path.join("dataset/baseline/saliency_basnet", name + ".png")).convert("L")) / 255

        pic = np.maximum(pic_1, pic_2)
        # pic = np.array(Image.open(os.path.join("data/cgl_dataset/salient_imgs_cgl",
        #                                       name)).convert("L").resize((513, 750))) / 255
        cal_mask = np.zeros_like(pic)

        cls = np.array(clses[idx], dtype=int)
        box = np.array(boxes[idx], dtype=int)

        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1

        total_area = np.sum(cal_mask)
        total_sal = np.sum(pic[cal_mask == 1])
        if total_sal and total_area:
            metrics += total_sal / total_area
    return metrics / len(img_names)


def main():
    import argparse
    from tqdm import tqdm

    # from html_to_ui import get_bbox, label_to_int
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str)
    args = parser.parse_args()
    filename = args.pred

    if filename.endswith(".json"):
        pred = read_json(filename)
    elif filename.endswith(".jsonl"):
        pred = read_jsonl(filename)

    BBOX = []
    CLS = []
    NAME = []
    CW = []
    CH = []

    for num, sample in enumerate(tqdm(pred)):
        idx = sample["id"]
        idx = idx.split("/")[-1]
        canvas_width = sample["canvas_width"]
        canvas_height = sample["canvas_height"]
        prediction = sample["predictions"][0]
        model_output = re.findall(r"##### .*? \$\$\$\$\$", prediction)
        bboxes = []
        clses = []

        try:
            for layer_id, item in enumerate(model_output):
                if layer_id in [0]:
                    continue
                pos = re.findall(r'"left": (.*?), "top": (.*?), "width": (.*?), "height": (.*?)[,}]', item)
                for p in pos:
                    bboxes.append([int(p[0]), int(p[1]), int(p[2]) + int(p[0]), int(p[3]) + int(p[1])])
                    clses.append(layer_id + 1)

            BBOX.append(bboxes)
            CLS.append(clses)
            NAME.append(idx)
            CW.append(canvas_width)
            CH.append(canvas_height)
        except:
            bboxes = []
            clses = []

    print("len:", len(NAME))
    val = metrics_val((0, 0), CLS, BBOX, CW, CH)
    print("metrics_val:", val)
    clses = getRidOfInvalid((0, 0), CLS, BBOX, CW, CH)
    ove = metrics_ove(CLS, BBOX)
    print("metrics_ove:", ove)
    ali = metrics_ali(CLS, BBOX, CW, CH)
    print("metrics_ali:", ali)
    und_l = metrics_und_l(CLS, BBOX)
    print("metrics_und_l:", und_l)
    und_s = metrics_und_s(CLS, BBOX)
    print("metrics_und_s:", und_s)
    # uti = metrics_uti(NAME, CLS, BBOX)
    # print("metrics_uti:", uti)
    # occ = metrics_occ(NAME, CLS, BBOX)
    # print("metrics_occ:", occ)
    # rea = metrics_rea(NAME, CLS, BBOX)
    # print("metrics_rea:", rea)


if __name__ == "__main__":
    main()
