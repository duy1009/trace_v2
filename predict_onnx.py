import imgproc
import cv2
import onnxruntime as rt 
import numpy as np
from postprocessor import getCellsAccordingToCornerEdge, adjustTableCoordinates

image_path = "/home/hungdv/tcgroup/Clone_project/table_reconstruct_api/img_test/Screenshot from 2024-06-08 11-36-27.png"
WEIGHT_PATH = "weights/trace_wtw.onnx"
CANVAS_SIZE = 1024
MAG_RATIO = 10
THRESHOLD1 = 0.2
THRESHOLD2 = 0.1
MAX_IMG_SIZE = 1024

sess = rt.InferenceSession(WEIGHT_PATH)
def predict_onnx(input, i_size = (3, 640, 640)):
    b, c, h, w = input.shape
    input = input.copy()
    i_size = list(i_size)
    i_size.insert(0, b)
    # input.resize(i_size)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    out = sess.run([label_name], {input_name: input.astype(np.float32)})[0]
    # out.resize((b, h//2, w//2, out.shape[3]))
    return out
def postproc(heatmap, threshold1, threshold2, target_ratio):
    # get params
    padx, pady = 0, 0
    mask_area = None
    if isinstance(target_ratio, int):
        ratio_w, ratio_h = 1 / target_ratio, 1 / target_ratio
    else:
        ratio_w, ratio_h = 1 / target_ratio[0], 1 / target_ratio[1]
        # get padding length
        if len(target_ratio) > 2:
            padx, pady = target_ratio[2], target_ratio[3]
        # get masking area
        if len(target_ratio) > 4:
            mask_area = target_ratio[4:8]  # x1, y1, x2, y2

    result = getCellsAccordingToCornerEdge(
        heatmap,
        threshold1,
        threshold2,
        mask_area=mask_area,
        transform_type=1,
    )
    result = adjustTableCoordinates(result, ratio_w, ratio_h, padx, pady)

    return result

def predict(image, canvas_size=1024, mag_ratio=10, threshold1 = 0.2, threshold2 = 0.1):
    s = canvas_size
    mag_ratio = mag_ratio
    image = imgproc.loadImage(image_path)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, s, mag_ratio=mag_ratio)
    input_size = img_resized.shape[:2]
    img_resized = cv2.resize(img_resized, (640, 640))
    x = imgproc.normalizeMeanVariance(img_resized)
    x = x.transpose((2, 0, 1))  # [h, w, c] to [c, h, w]
    x = np.expand_dims(x, axis=0)  # [c, h, w] to [b, c, h, w]
    out = predict_onnx(x)
    res_heatmap = out[0].copy()
    res_heatmap = cv2.resize(res_heatmap,(input_size[1]//2, input_size[0]//2))
    result = postproc(res_heatmap, threshold1, threshold2, target_ratio)
    return result

def infer(image):
    shape_max = np.argmax(image.shape)
    result, res_score = predict(image, CANVAS_SIZE, MAG_RATIO, THRESHOLD1, THRESHOLD2)

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
        table_xywh = [int(x) for x in table_xywh]
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
            quad = [int(x) for x in cell["quad"]]
            x1, y1, x2, y2, x3, y3, x4, y4 = quad
            bbox_cell = f"[{x1},{y1},{x4},{y4},{x3},{y3},{x2},{y2}]"
            html = f"<td colspan=\"{span_col}\" rowspan=\"{span_row}\">{bbox_cell}</td>"
            html_fels[cell['row_range'][0]].append(html)
        for hf in html_fels:
            data+="<tr>" + " ".join(hf) + "</tr>\n"
        return data