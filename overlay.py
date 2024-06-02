import argparse
import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import cv2

# load a pre-trained mask r-cnn model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# get predictions from the model
def get_predictions(img, threshold=0.5):
    transform = F.to_tensor(img)
    img_tensor = transform.unsqueeze(0)
    
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

# apply the image overlay on the detected object masks
def apply_bsod_overlay(img_cv, masks, overlay_image, alpha=0.6):
    overlay_resized = cv2.resize(overlay_image, (img_cv.shape[1], img_cv.shape[0]))

    for mask in masks:
        mask = mask[0]  # remove extra dimension
        mask = (mask > 0.5).astype(np.uint8)
        for c in range(0, 3):
            img_cv[:, :, c] = np.where(
                mask == 1,
                cv2.addWeighted(overlay_resized[:, :, c], alpha, img_cv[:, :, c], 1 - alpha, 0),
                img_cv[:, :, c]
            )
    return img_cv

# main function to process images
def main(args):
    # load an image from the specified path
    img = Image.open(args.input_image)

    # load the overlay image
    overlay_image = cv2.imread(args.overlay_image)

    # get predictions
    labels, boxes, masks = get_predictions(img)

    # coco class names for mask r-cnn
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
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # get the corresponding label index for the object
    if args.object_to_detect in COCO_INSTANCE_CATEGORY_NAMES:
        object_label_index = COCO_INSTANCE_CATEGORY_NAMES.index(args.object_to_detect)
    else:
        raise ValueError(f"{args.object_to_detect} is not in the coco dataset.")

    # filter masks for the desired object
    filtered_masks = [mask for label, mask in zip(labels, masks) if label == object_label_index]

    # convert pil image to opencv format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # apply the overlay effect with transparency and save the result
    if len(filtered_masks) > 0:
        img_with_effect = apply_bsod_overlay(img_cv, filtered_masks, overlay_image, alpha=0.6)

        # create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "output_image.png")

        cv2.imwrite(output_path, img_with_effect)
        print(f"image saved as {output_path}")
    else:
        print(f"no {args.object_to_detect} detected")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply BSOD overlay effect to detected objects in an image.")
    parser.add_argument('--input_image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--overlay_image', type=str, required=True, help="Path to the BSOD overlay image.")
    parser.add_argument('--object_to_detect', type=str, required=True, help="Object to detect in the image.")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save the output image.")
    
    args = parser.parse_args()
    main(args)
