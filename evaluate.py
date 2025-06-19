import cv2
import xml.etree.ElementTree as ET
import os

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union

def get_ground_truth_boxes(xml_path, target_class='cat'):
    """Extract bounding boxes from Pascal VOC XML"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    boxes = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == target_class:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])  # [x, y, w, h]
    
    return filename, boxes

# Configuration
cascade_path = "cat_detector/cascade.xml"
dataset_root = "dataset"
target_class = "cat" 

# Load cascade
cascade = cv2.CascadeClassifier(cascade_path)
if cascade.empty():
    print("Error: Could not load cascade classifier")
    exit()

# Get all XML files
xml_dir = os.path.join(dataset_root, 'annotations', 'xmls')
image_dir = os.path.join(dataset_root, 'images')
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

# Count correct detections
total_ground_truth = 0
total_detections = 0
correct_detections = 0

print(f"Testing {len(xml_files)} images...")

for xml_file in xml_files:
    xml_path = os.path.join(xml_dir, xml_file)
    filename, gt_boxes = get_ground_truth_boxes(xml_path, target_class)
    
    # Skip if no ground truth objects
    if len(gt_boxes) == 0:
        continue
    
    # Load and detect
    image_path = os.path.join(image_dir, filename)
    if not os.path.exists(image_path):
        continue
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect objects
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=30)
    
    total_ground_truth += len(gt_boxes)
    total_detections += len(detections)
    
    # Check which detections are correct (IoU > 0.5)
    for detection in detections:
        for gt_box in gt_boxes:
            if calculate_iou(detection, gt_box) > 0.5:
                correct_detections += 1
                break

# Calculate accuracy
precision = correct_detections / total_detections if total_detections > 0 else 0
recall = correct_detections / total_ground_truth if total_ground_truth > 0 else 0

print(f"\nResults:")
print(f"Ground truth objects: {total_ground_truth}")
print(f"Total detections: {total_detections}")
print(f"Correct detections: {correct_detections}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")

