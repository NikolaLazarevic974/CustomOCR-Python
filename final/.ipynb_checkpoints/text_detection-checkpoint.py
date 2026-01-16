import sys
sys.path.append('../CRAFT-pytorch')
from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates

import torch
from collections import OrderedDict

import cv2

import numpy as np

import matplotlib.pyplot as plt

def load_craft():
    model = CRAFT()
    state_dict = torch.load('../CRAFT-pytorch/craft_mlt_25k.pth', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.` prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    return model


def extract_features(binary_image, craft_model):
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB) 
    binary_image = binary_image.astype(np.float32) / 255.0
    binary_image = torch.from_numpy(binary_image).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    with torch.no_grad():
        y, feature = craft_model(binary_image)
    return y, feature
def get_boxes(y):
    score_text = y[0,:,:,0].cpu().numpy()
    score_link = y[0,:,:,1].cpu().numpy()
    
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold=0.7, link_threshold=0.4, low_text=0.4)
    boxes = adjustResultCoordinates(boxes, ratio_w=1.0, ratio_h=1.0)
    
    return boxes

def visualize_text_detection(binary_image, boxes):

    result_image = binary_image.copy()
    for box in boxes:
        box = box.astype(int)
        cv2.polylines(result_image, [box], isClosed=True, color=(0,0,255), thickness=2)
    
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def extract_box_images(binary_image, boxes):
    result = []
    for i, box in enumerate(boxes):
        box = box.astype(int)
        
        x, y, w, h = cv2.boundingRect(box)

        cropped = binary_image[y:y+h, x:x+w]

        result.append(cropped)
    return cropped

def sort_boxes(boxes, row_tolerance=20):
    """
    Sort text boxes in reading order (top-to-bottom, left-to-right).
    
    Args:
        boxes: List of boxes, where each box is array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        row_tolerance: Vertical tolerance in pixels to consider boxes in the same row
        
    Returns:
        Sorted list of boxes in reading order
    """
    if len(boxes) == 0:
        return boxes
    
    # Calculate center point and top-left corner for each box
    box_info = []
    for box in boxes:
        # Get bounding box coordinates
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_center = sum(x_coords) / 4
        y_center = sum(y_coords) / 4
        
        box_info.append({
            'box': box,
            'x_min': x_min,
            'y_min': y_min,
            'x_center': x_center,
            'y_center': y_center
        })
    
    # Sort by y_center first (top to bottom)
    box_info.sort(key=lambda b: b['y_center'])
    
    # Group boxes into rows based on row_tolerance
    rows = []
    current_row = [box_info[0]]
    
    for i in range(1, len(box_info)):
        # Check if current box is in the same row as previous box
        if abs(box_info[i]['y_center'] - current_row[-1]['y_center']) <= row_tolerance:
            current_row.append(box_info[i])
        else:
            # Sort current row by x position (left to right)
            current_row.sort(key=lambda b: b['x_min'])
            rows.append(current_row)
            current_row = [box_info[i]]
    
    # Don't forget the last row
    current_row.sort(key=lambda b: b['x_min'])
    rows.append(current_row)
    
    # Flatten rows back into a single list
    sorted_boxes = []
    for row in rows:
        sorted_boxes.extend([b['box'] for b in row])
    
    return sorted_boxes





