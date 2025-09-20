# image_utils.py

import numpy as np
import random
import colorsys
import matplotlib
# Use a non-interactive backend to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import cv2
from PIL import Image
import traceback

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c],
        )
    return image

def display_instances(
    image,
    boxes,
    masks,
    class_ids,
    class_names,
    output_file='image_boxed.png',
    scores=None,
    show_mask=False,
    show_bbox=True,
    colors=None,
    captions=None,
    dpi=300,
):
    """
    Display instances on the image.
    """
    try:
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            # Create a blank image
            height, width = image.shape[:2]
            blank_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.imwrite(output_file, blank_image)
            return

        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        
        # Create figure with proper dimensions
        height, width = image.shape[:2]
        fig, ax = plt.subplots(1, figsize=(16, 16))
        
        # Use red color for all boxes
        colors = [(1, 0, 0)] * N  # Red color in RGB (1, 0, 0)

        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis("off")

        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]

            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            
            if show_bbox:
                p = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    alpha=0.7,
                    linestyle="dashed",  # Use dashed lines
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(p)

            if not captions:
                class_id = class_ids[i]
                label = class_names[class_id]
                caption = label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

            mask = masks[:, :, i]
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)

        ax.imshow(masked_image.astype(np.uint8))
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Successfully saved boxed image using matplotlib: {output_file}")
    except Exception as e:
        print(f"Error in display_instances: {e}")
        # Re-raise to let the calling function handle it with fallback
        raise

def save_box_image(bboxes, masks, idx, page, output="image_boxed.png"):
    """
    Save an image with bounding boxes and masks applied.
    """
    try:
        # Add some debugging information
        print(f"Saving boxed image {output}")
        print(f"  Page shape: {page.shape}")
        print(f"  Number of bboxes: {len(bboxes)}")
        print(f"  Number of masks: {masks.shape[2] if len(masks.shape) > 2 else 0}")
        print(f"  Requested index: {idx}")
        
        if idx >= len(bboxes):
            print(f"Error: Index {idx} is out of range for bboxes array of length {len(bboxes)}")
            # Create a simple image with a message
            height, width = page.shape[:2]
            blank_image = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(blank_image, f"Error: Invalid bbox index {idx}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(output, blank_image)
            return
            
        bbox = bboxes[idx]
        print(f"  Bbox coordinates: {bbox}")
        
        # Validate bounding box coordinates
        height, width = page.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
            print(f"  Warning: Bbox coordinates out of bounds. Page dimensions: {width}x{height}")
            # Clamp coordinates to valid range
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
        
        bboxs2 = bboxes[idx:idx+1]
        masks2 = masks[:,:,idx:idx+1]

        display_instances(
            image=page,
            masks=masks2,
            class_ids=np.array([0]),
            boxes=np.array(bboxs2),
            class_names=np.array(["structure"]),
            output_file=output
        )
    except Exception as e:
        print(f"Warning: Failed to save boxed image using matplotlib {output}: {e}")
        # Use a reliable fallback method with actual bounding box coordinates
        try:
            page_copy = page.copy()
            bbox = bboxes[idx]
            x1, y1, x2, y2 = map(int, bbox)
            # 使用红色虚线框
            cv2.rectangle(page_copy, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(output, page_copy)
            print(f"Used fallback method to save boxed image {output}")
        except Exception as fallback_e:
            print(f"Error: Failed to save boxed image even with fallback method: {fallback_e}")
            # Create a blank image with error message
            try:
                height, width = page.shape[:2]
                blank_image = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(blank_image, "Error: Could not save boxed image", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(output, blank_image)
            except Exception as final_e:
                print(f"Final error: Could not create error image: {final_e}")
