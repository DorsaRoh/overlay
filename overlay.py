import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2

IMAGE_PATH= "input/girl2.jpg"
OVERLAY_PATH = "overlay/bsod.png"
OBJECT = 'person'
output_path = "output/output_image.png"

# Load a pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to get predictions from the model
def get_predictions(img, threshold=0.5):
    transform = F.to_tensor(img)
    img_tensor = transform.unsqueeze(0)

    # Ensure the image has 3 channels
    if img_tensor.shape[1] == 4:
        img_tensor = img_tensor[:, :3, :, :]

    # Normalize the image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_tensor = F.normalize(img_tensor, mean=mean, std=std)

    with torch.no_grad():
        predictions = model(img_tensor)

    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy()

    pred_labels = pred_labels[pred_scores > threshold]
    pred_boxes = pred_boxes[pred_scores > threshold]
    pred_masks = pred_masks[pred_scores > threshold]

    return pred_labels, pred_boxes, pred_masks

# Function to apply the BSOD image overlay on the detected object masks
def apply_bsod_overlay(img_cv, masks, overlay_image, alpha=0.6):
    overlay_resized = cv2.resize(overlay_image, (img_cv.shape[1], img_cv.shape[0]))

    for mask in masks:
        mask = mask[0]  # Remove extra dimension
        mask = (mask > 0.5).astype(np.uint8)
        for c in range(0, 3):
            img_cv[:, :, c] = np.where(
                mask == 1,
                cv2.addWeighted(overlay_resized[:, :, c], alpha, img_cv[:, :, c], 1 - alpha, 0),
                img_cv[:, :, c]
            )
    return img_cv

# User input for the object to detect
object_to_detect = OBJECT  # Replace with desired object

# Load an image from a relative path
image_path = IMAGE_PATH  # Replace with your image path
img = Image.open(image_path)

# Load the BSOD overlay image
overlay_path = OVERLAY_PATH  # Replace with your BSOD image path
overlay_image = cv2.imread(overlay_path)

# Get predictions
labels, boxes, masks = get_predictions(img)

# COCO class names for Mask R-CNN
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Get the corresponding label index for the object
if object_to_detect in COCO_INSTANCE_CATEGORY_NAMES:
    object_label_index = COCO_INSTANCE_CATEGORY_NAMES.index(object_to_detect)
else:
    raise ValueError(f"{object_to_detect} is not in the COCO dataset.")

# Filter masks for the desired object
filtered_masks = [mask for label, mask in zip(labels, masks) if label == object_label_index]

# Convert PIL image to OpenCV format
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Apply the overlay effect with transparency and save the result
if len(filtered_masks) > 0:
    img_with_effect = apply_bsod_overlay(img_cv, filtered_masks, overlay_image, alpha=0.8)
    cv2.imwrite(output_path, img_with_effect)
    print(f"Image saved as {output_path}")
else:
    print(f"No {object_to_detect} detected")

# Display the image
img_result = Image.open(output_path)
img_result.show()
