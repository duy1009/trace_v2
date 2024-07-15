# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import imgproc
import postprocessor
from model import TraceModel


net = TraceModel()
CUDA = torch.cuda.is_available()
WEIGHT = "./weights/trace_wtw.pth"
CANVAS_SIZE = 1024
MAG_RATIO = 10
THRESHOLD1 = 0.2
THRESHOLD2 = 0.1
MAX_IMG_SIZE = 1024
PREDICT_SQUARE = True
SQUARE_SIZE = 640

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def predict(net, image, canvas_size=1024, mag_ratio=10, cuda=True, threshold1 = 0.2, threshold2 = 0.1):

    # resize
    s = canvas_size
    mag_ratio = mag_ratio
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, s, mag_ratio=mag_ratio)

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y = y[0]  # ignore feature map
    res_heatmap = y[0].cpu().data.numpy()

    # Post-processing
    result = postprocessor.run2(res_heatmap, threshold1, threshold2, target_ratio)

    return result, res_heatmap

def predictSquare(net, image, canvas_size=1024, mag_ratio=10, cuda=True, threshold1 = 0.2, threshold2 = 0.1, size = 640):

    # resize
    s = canvas_size
    mag_ratio = mag_ratio
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, s, mag_ratio=mag_ratio)
    input_size = img_resized.shape[:2]
    img_resized = cv2.resize(img_resized, (size, size))
    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y = y[0]  # ignore feature map
    res_heatmap = y[0].cpu().data.numpy()
    res_heatmap = cv2.resize(res_heatmap,(input_size[1]//2, input_size[0]//2))

    # Post-processing
    result = postprocessor.run2(res_heatmap, threshold1, threshold2, target_ratio)

    return result, res_heatmap

def infer(image):
    shape_max = np.argmax(image.shape)
    if not PREDICT_SQUARE and image.shape[shape_max] > MAX_IMG_SIZE:
        scale = MAX_IMG_SIZE/image.shape[shape_max]
    else:
        scale = 1
    image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))
    
    if PREDICT_SQUARE:
        result, res_score = predictSquare(net, image, CANVAS_SIZE, MAG_RATIO, CUDA, THRESHOLD1, THRESHOLD2, SQUARE_SIZE)
    else:
        result, res_score = predict(net, image, CANVAS_SIZE, MAG_RATIO, CUDA, THRESHOLD1, THRESHOLD2)

    data = r"""
    <html>
        <head> <meta charset="UTF-8">
        <style>
        table, th, td {
            border: 1px solid black;
            font-size: 10px;
        }
        </style> </head>
        <body>
        <table frame="hsides" rules="groups" width="100%%" >
        """
    for trace_res in result:
        table_xywh = trace_res["rect"]
        table_xywh = [int(x/scale) for x in table_xywh]
        table_quad = [
            [table_xywh[0], table_xywh[1]],
            [table_xywh[0] + table_xywh[2], table_xywh[1]],
            [table_xywh[0] + table_xywh[2], table_xywh[1] + table_xywh[3]],
            [table_xywh[0], table_xywh[1] + table_xywh[3]],
        ]

        bbox_table = f"{table_quad[0][0]},{table_quad[0][1]},{table_quad[3][0]},{table_quad[3][1]},{table_quad[2][0]},{table_quad[2][1]},{table_quad[1][0]},{table_quad[1][1]}"
        nrow = 0
        cnt =0
        for cell in trace_res["cells"]:
            if cnt!=cell['row_range'][1] and cnt >nrow:
                nrow = cnt
            cnt = cell['row_range'][1]
        nrow+=1
        html_fels = [[] for i in range(nrow)]
        for cell in trace_res["cells"]:
            sr = cell['row_range'][0]
            er = cell['row_range'][1]
            sc = cell['col_range'][0]
            ec = cell['col_range'][1]
            span_row = er - sr +1
            span_col = ec - sc +1
            quad = [int(x/scale) for x in cell["quad"]]
            x1, y1, x2, y2, x3, y3, x4, y4 = quad
            bbox_cell = f"[{x1},{y1},{x4},{y4},{x3},{y3},{x2},{y2}]"
            html = f"<td colspan=\"{span_col}\" rowspan=\"{span_row}\">{bbox_cell}</td>"
            html_fels[cell['row_range'][0]].append(html)
        for hf in html_fels:
            data+="<tr>" + " ".join(hf) + "</tr>\n"
        return data

if CUDA:
    net = net.cuda()
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

if CUDA:
    net.load_state_dict(torch.load(WEIGHT))
else:
    net.load_state_dict(copyStateDict(torch.load(WEIGHT, map_location="cpu")))
net.eval()