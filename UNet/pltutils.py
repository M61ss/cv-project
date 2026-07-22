import cv2

def apply_mask(img, mask):
    img_rgb = cv2.cvtColor(img.numpy(), cv2.COLOR_GRAY2RGB)

    overlay = img_rgb.copy()
    overlay[mask == 1] = [100, 150, 255]
    overlay[mask == 2] = [100, 255, 100]
    overlay[mask == 3] = [255, 100, 100]
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    return blended