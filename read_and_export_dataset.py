import pathlib
import xml.etree.ElementTree as ET
import numpy as np

def parse_xml(xmlfile: str) -> dict:
    root = ET.parse(xmlfile).getroot()
    return {
        "filename": root.find("filename").text,
        "objects": [{
            "name": box.find('name').text,
            "bbox": {
                "xmin": int(bb.find("xmin").text),
                "ymin": int(bb.find("ymin").text),
                "xmax": int(bb.find("xmax").text),
                "ymax": int(bb.find("ymax").text)
            }
        } for box in root.iter('object') for bb in [box.find('bndbox')]]
    }

def format_bbox(obj):
    x, y = obj["bbox"]["xmin"], obj["bbox"]["ymin"]
    w, h = obj["bbox"]["xmax"] - x, obj["bbox"]["ymax"] - y
    return f"{x} {y} {w} {h}"

# Read Pascal VOC and write data
base_path = pathlib.Path("dataset")
img_src = base_path / "images"
ann_src = base_path / "annotations" / "xmls"

negative, positive = [], []
for xmlfile in ann_src.glob("*.xml"):
    ann = parse_xml(str(xmlfile))
    img_path = str(img_src / ann['filename'])
    
    if ann['objects'][0]['name'] == 'dog':
        negative.append(img_path)
    elif ann['objects'][0]['name'] == 'cat':
        bboxes = [format_bbox(obj) for obj in ann['objects'] if obj['name'] == 'cat']
        if bboxes:
            positive.append(f"{img_path} {len(bboxes)} {' '.join(bboxes)}")

pathlib.Path("negative.dat").write_text("\n".join(negative))
pathlib.Path("positive.dat").write_text("\n".join(positive))