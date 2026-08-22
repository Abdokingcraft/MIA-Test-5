import os
import glob
import cv2
import numpy as np


def get_masks(resized):
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernelb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    lower_blue = np.array([0, 80, 100])
    upper_blue = np.array([17, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    lower_red = np.array([108, 100, 95])
    upper_red = np.array([140, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernelb, iterations=1)

    return red_mask, blue_mask


def extract_ball(mask, resized, area_fraction=0.18, label="Ball"):
    total_pixels = cv2.countNonZero(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    best_score = 0
    for c in contours:
        area = cv2.contourArea(c)
        if label == "Blue Ball":
            area_fraction = 0.12
        if total_pixels == 0 or area < area_fraction * total_pixels:
            continue
        M = cv2.moments(c)
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        pts = c.reshape(-1, 2)
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        roundness = 1 - (np.std(dists) / np.mean(dists))
        if roundness > best_score:
            best_score = roundness
            best_contour = c
    if best_contour is None:
        print(f"No {label.lower()} found")
        return None, None, None

    ball_mask = np.zeros_like(mask)
    cv2.drawContours(ball_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
    ball_only = cv2.bitwise_and(resized, resized, mask=ball_mask)
    ball_gray = cv2.cvtColor(ball_only, cv2.COLOR_BGR2GRAY)
    bbox = cv2.boundingRect(best_contour)

    return ball_only, ball_gray, bbox


def detect_circle(ball_gray, label="Ball", min=10):
    if ball_gray is None:
        print(f"No {label.lower()} to detect a circle in")
        return None

    if label == "Red Ball":
        param1 = 20
    elif label == "Blue Ball":
        param1 = 35

    blurred = cv2.GaussianBlur(ball_gray, (9, 9), 25)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=param1, param2=20, minRadius=min, maxRadius=10000)

    if circles is None:
        print(f"No circle found for {label}")
        return None

    circles = np.round(circles[0, :]).astype(int)
    x, y, r = circles[0]

    return (x, y, r)


def bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    return x_center, y_center, norm_w, norm_h


def write_label_file(labels, output_path):
    with open(output_path, "w") as f:
        for class_id, x_c, y_c, w, h in labels:
            f.write(f"{class_id} {x_c:.2f} {y_c:.2f} {w:.2f} {h:.2f}\n")


def process_folder(images_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
    )

    for image_path in image_paths:
        bgr = cv2.imread(image_path)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        resized = img

        red_mask, blue_mask = get_masks(resized)

        _, red_ball_gray, red_bbox = extract_ball(red_mask, resized, label="Red Ball")
        _, blue_ball_gray, blue_bbox = extract_ball(blue_mask, resized, label="Blue Ball")

        red_circle = detect_circle(red_ball_gray, label="Red Ball")
        blue_circle = detect_circle(blue_ball_gray, label="Blue Ball")

        red_r = red_circle[2] if red_circle is not None else None
        blue_r = blue_circle[2] if blue_circle is not None else None

        if red_r is not None and blue_r is not None:
            if red_r < blue_r / 2:
                red_bbox = None
            elif blue_r < red_r / 2:
                blue_bbox = None

        labels = []
        if red_bbox is not None:
            x_c, y_c, bw, bh = bbox_to_yolo(red_bbox, w, h)
            labels.append((1, x_c, y_c, bw, bh))
        if blue_bbox is not None:
            x_c, y_c, bw, bh = bbox_to_yolo(blue_bbox, w, h)
            labels.append((0, x_c, y_c, bw, bh))

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, base_name + ".txt")
        write_label_file(labels, output_path)

        print(f"{os.path.basename(image_path)}: {len(labels)} ball(s) detected -> {output_path}")


if __name__ == "__main__":
    IMAGES_DIR = "balls/"
    OUTPUT_DIR = "labels/"

    process_folder(IMAGES_DIR, OUTPUT_DIR)