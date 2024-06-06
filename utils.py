from ultralytics import YOLO



import re

MDP = "LaliloP2IA2024"

def remove_overset(latex_str):
    # Define regex pattern to match overset commands
    overset_pattern = r'\\overset\s*{[^{}]*}'
    
    # Remove overset commands from the LaTeX string
    cleaned_latex = re.sub(overset_pattern, '', latex_str)
    
    return cleaned_latex

def remove_latting_0(latex_str):
    pattern = r'\b0+(\d+)'
    result = re.sub(pattern, r'\1', latex_str)
    return result


def clean_latex_expression(expression):
    cleaned_expression = remove_overset(expression.replace(" ", "")).replace("{", "").replace("}", "").replace('=', '==')
    return remove_latting_0(cleaned_expression)


model = YOLO("models/yolov8_h3.pt")


def get_bounding_boxes_yolov8(img_path):
    detections = model(img_path)
    confs = detections[0].boxes.conf
    classes = detections[0].boxes.cls
    boxes = detections[0].boxes.xyxy
    conf_thr = 0.0
    bounding_boxes = []
    for elem in zip(boxes, classes, confs):
        top_left = (int(elem[0][0]), int(elem[0][1]))
        bottom_right = (int(elem[0][2]), int(elem[0][3]))
        label = str(int(elem[1]))
        conf = float(elem[2])
        # Convert int value labels to their corresponding classes:
        if label == "10":
            label = "+"
        elif label == "11":
            label = "-"
        elif label == "12":
            label = "="
        elif label == "13":
            label = "r1"
        # Filter low-confidence detections:
        if conf > conf_thr:
            bounding_boxes.append(([top_left, bottom_right], label, conf))
    return bounding_boxes
