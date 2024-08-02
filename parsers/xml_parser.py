import os
import random, tqdm
from collections.abc import Iterable
from xml.dom import minidom

import file_utils


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


class ParserTRACE:
    def __init__(self, root_path, dataset, phase):
        self.gt = []
        base_folder = os.path.join(root_path, dataset)
        base_folder = os.path.join(base_folder, phase)
        image_files, _, _ = file_utils.list_files(base_folder)
        for img_file in tqdm.tqdm(image_files):
            basename, ext = os.path.splitext(os.path.basename(img_file))
            gt_file = os.path.join(base_folder,"xml", f"{basename}.xml")
            if os.path.exists(gt_file):
                quads = []
                lines = []
                with open(gt_file, "r") as f:
                    dom = minidom.parse(f)

                elem_tables = dom.documentElement.getElementsByTagName("table")
                for e_table in elem_tables:
                    elem_cells = e_table.getElementsByTagName("cell")
                    for e_cell in elem_cells:
                        points = str(e_cell.getElementsByTagName("Coords")[0].getAttribute("points"))
                        new_points = []
                        for p in points.split():
                            new_points.append(p.split(","))
                        quad = list(flatten(new_points))
                        quads.append(quad)

                        line_t = bool(int(e_cell.getElementsByTagName("Lines")[0].getAttribute("top")))
                        line_b = bool(int(e_cell.getElementsByTagName("Lines")[0].getAttribute("bottom")))
                        line_l = bool(int(e_cell.getElementsByTagName("Lines")[0].getAttribute("left")))
                        line_r = bool(int(e_cell.getElementsByTagName("Lines")[0].getAttribute("right")))
                        line = [line_t, line_b, line_l, line_r]
                        lines.append(line)

                self.gt.append({"file_name": img_file, "quads": quads, "lines": lines})

    def getDatasetSize(self):
        return len(self.gt)

    def lenFiles(self):
        return len(self.gt)

    def parseGT(self, index=-1):
        if index == -1:
            gt = self.gt[random.randrange(0, len(self.gt))]
        else:
            gt = self.gt[index]

        return gt["file_name"], gt

import xml.etree.ElementTree as ET

class ParserTRACE_wtw:
    def __init__(self, root_path, dataset, phase):
        self.gt = []
        base_folder = os.path.join(root_path, dataset)
        base_folder = os.path.join(base_folder, phase)
        image_files, _, _ = file_utils.list_files(base_folder)
        for img_file in tqdm.tqdm(image_files):
            basename, ext = os.path.splitext(os.path.basename(img_file))
            if phase=="train":
                gt_file = os.path.join(base_folder,"xml", f"{basename}..xml")
            else:
                gt_file = os.path.join(base_folder,"xml", f"{basename}.xml")
            if os.path.exists(gt_file):
                quads = []
                lines = []
                with open(gt_file, 'r') as f:
                    xml_data = f.read()
                root = ET.fromstring(xml_data)
                for object_elem in root.findall('.//object'):
                    bndbox_elem = object_elem.find('bndbox')
                    x1 = float(bndbox_elem.find('x1').text)
                    y1 = float(bndbox_elem.find('y1').text)
                    x2 = float(bndbox_elem.find('x2').text)
                    y2 = float(bndbox_elem.find('y2').text)
                    x3 = float(bndbox_elem.find('x3').text)
                    y3 = float(bndbox_elem.find('y3').text)
                    x4 = float(bndbox_elem.find('x4').text)
                    y4 = float(bndbox_elem.find('y4').text)
                    quads.append([x1, y1, x2, y2, x3, y3, x4, y4])
                    lines.append([1, 1, 1, 1])

                self.gt.append({"file_name": img_file, "quads": quads, "lines": lines})

    def getDatasetSize(self):
        return len(self.gt)

    def lenFiles(self):
        return len(self.gt)

    def parseGT(self, index=-1):
        if index == -1:
            gt = self.gt[random.randrange(0, len(self.gt))]
        else:
            gt = self.gt[index]

        return gt["file_name"], gt
if __name__ == "__main__":
    parser = ParserTRACE_wtw("/home/hungdv/tcgroup/dataset/tabelSeg/wtw/test_code_train", "train", "train")
    print(parser.lenFiles())
    print(parser.parseGT(-1))

