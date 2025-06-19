import cv2
from pathlib import Path

def load_image(image_path: str) -> tuple:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def detect_objects(classifier, gray_img, scale=1.1, neighbors=30, min_size=(30, 30)):
    return classifier.detectMultiScale(
        gray_img,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=min_size
    )

def draw_detections(img, detections, color=(255, 0, 0), thickness=2):
    for (x, y, w, h) in detections:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
    return img

def main():
    image_path = 'dataset/images/shiba_inu_64.jpg'
    model_path = 'dog_detector/cascade.xml'
    
    classifier = cv2.CascadeClassifier(model_path)
    img, gray = load_image(image_path)
    
    detections = detect_objects(classifier, gray)
    result = draw_detections(img, detections)
    
    cv2.imshow('Object Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()