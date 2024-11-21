# image_utils.py

import numpy as np
import random
import colorsys
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import cv2
from PIL import Image

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
    show_mask=True,
    show_bbox=True,
    colors=None,
    captions=None,
    dpi=300,
):
    """
    Display instances on the image.
    """
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
        return

    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    _, ax = plt.subplots(1, figsize=(16, 16))

    colors = colors or random_colors(N)

    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
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

        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8
        )
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_box_image(bboxes, masks, idx, page, output="image_boxed.png"):
    """
    Save an image with bounding boxes and masks applied.
    """
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
