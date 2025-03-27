import torch
import open_clip
from PIL import Image
import cv2
import numpy as np

def clip_verification(image_path, threshold=0.25):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    image = preprocess(Image.open(image_path)).unsqueeze(0)
    texts = ["a football pitch", "a close-up of a football player", "a stadium with crowd", "a training ground", "a grass field"]

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_tokens = tokenizer(texts)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_prob, top_label = similarity[0].max(0)

    return texts[top_label] == "a football pitch" and top_prob.item() > threshold

def is_close_plan(mask_green, threshold=0.8, band_ratio=0.025):
    h, w = mask_green.shape
    band_h = int(h * band_ratio)
    band_w = int(w * band_ratio)

    top = mask_green[:band_h, :]
    bottom = mask_green[-band_h:, :]
    left = mask_green[:, :band_w]
    right = mask_green[:, -band_w:]

    green_ratios = [
        np.sum(top > 0) / top.size,
        np.sum(bottom > 0) / bottom.size,
        np.sum(left > 0) / left.size,
        np.sum(right > 0) / right.size
    ]

    return all(ratio > threshold for ratio in green_ratios)

def detect_goal_net_by_lines(lines):
    if lines is None or len(lines) < 15:
        return False

    vertical_lines = 0
    horizontal_lines = 0
    line_lengths = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
        length = np.linalg.norm([x2 - x1, y2 - y1])

        if angle < 10:
            horizontal_lines += 1
        elif angle > 80:
            vertical_lines += 1

        line_lengths.append(length)

    avg_length = np.mean(line_lengths)
    std_length = np.std(line_lengths)

    is_grid = vertical_lines > 15 and horizontal_lines > 15
    is_short_lines = avg_length < 50
    is_uniform_length = std_length < 10

    return (is_grid and is_short_lines) or (is_grid and is_uniform_length)

def detect_pitch(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((10, 10), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            mask_obj = np.zeros_like(mask_green)
            cv2.drawContours(mask_obj, [cnt], -1, 255, thickness=cv2.FILLED)

            green_pixels = np.sum((mask_green > 0) & (mask_obj > 0))
            total_pixels = np.sum(mask_obj > 0)

            green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0
            if green_ratio < 0.5:
                cv2.drawContours(mask_cleaned, [cnt], -1, 0, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_cleaned)
    edges = cv2.Canny(masked_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/360, threshold=50, minLineLength=50, maxLineGap=8)

    if detect_goal_net_by_lines(lines):
        lines = None

    green_ratio = np.sum(mask_cleaned > 0) / mask_cleaned.size
    total_line_length = sum(np.linalg.norm([x2 - x1, y2 - y1]) for x1, y1, x2, y2 in lines[:, 0]) if lines is not None else 0

    score = 0.3 * green_ratio + 0.7 * (total_line_length / 4500)
    score = min(1, score)

    if is_close_plan(mask_green, threshold=0.7):
        return 0

    if 0.8 <= score < 1.0:
        is_pitch = clip_verification(image_path)
        return 1 if is_pitch else 0

    return score
