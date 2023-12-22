from PIL.Image import alpha_composite
import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import floor

def get_matches(fresque, fragment):
    fresque_rgb = cv2.cvtColor(fresque, cv2.COLOR_BGR2RGB)
    fragment_rgb = cv2.cvtColor(fragment, cv2.COLOR_BGR2RGB)

    fresque_gray = cv2.cvtColor(fresque_rgb, cv2.COLOR_RGB2GRAY)
    fragment_gray = cv2.cvtColor(fragment_rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    fresque_keypoints, des1 = sift.detectAndCompute(fresque_gray, None)
    fragment_keypoints, des2 = sift.detectAndCompute(fragment_gray, None)

    if not fresque_keypoints or not fragment_keypoints:
        return -1, -1, -1

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    fresque_matches = []
    fragment_matches = []

    if matches:
        for match in matches:
            if match.distance < 500:
                fresque_matches.append(fresque_keypoints[match.queryIdx].pt)
                fragment_matches.append(fragment_keypoints[match.trainIdx].pt)

    if len(fresque_matches) > 1:
        reference_point = fresque_matches[0]
        fresque_matches, fragment_matches = filter_matches(fresque_matches, fragment_matches, reference_point)

        angle = angle_between_vectors(
            np.array(fresque_matches[0]) - np.array(fresque_matches[1]),
            np.array(fragment_matches[0]) - np.array(fragment_matches[1])
        )

        offset_x, offset_y = distance_between_points(fragment.shape[1] / 2, fragment_matches[0][0],
                                                      fragment.shape[0] / 2, fragment_matches[0][1])
        v = rotate_vector((offset_x, offset_y), angle)

        return fresque_matches[0][0] - v[0], fresque_matches[0][1] - v[1], -angle

    elif len(fresque_matches) == 1:
        return fresque_matches[0][0], fresque_matches[0][1], 0

    else:
        return -1, -1, -1

def filter_matches(fresque_matches, fragment_matches, reference_point):
    filtered_fresque_matches, filtered_fragment_matches = [], []

    for match, fragment_match in zip(fresque_matches, fragment_matches):
        if euclidean_distance(reference_point[0], reference_point[1], match[0], match[1]) <= 500:
            filtered_fresque_matches.append(match)
            filtered_fragment_matches.append(fragment_match)

    return filtered_fresque_matches, filtered_fragment_matches

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def distance_between_points(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    return x, y

def rotate_vector(vector, angle):
    angle = np.deg2rad(angle)
    return np.array([vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
                     vector[0] * np.sin(angle) + vector[1] * np.cos(angle)])

def angle_between_vectors(v1, v2):
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)) * 180 / np.pi

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def add_overlay(fresque, fragment, x, y, angle):
    fragment = rotate_image(fragment, angle)
    b, g, r, a = cv2.split(fragment)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = fresque[y:y + h, x:x + w]

    fragment1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    fragment2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    fresque[y:y + h, x:x + w] = cv2.add(fragment1_bg, fragment2_fg)

    return fragment

def write_file(fragment_positions, filename):
    with open(filename, "w") as f:
        for i, x, y, angle in fragment_positions:
            f.write(f"{i} {x} {y} {angle}\n")

def main():
    fresque = cv2.imread('Michelangelo/Michelangelo_ThecreationofAdam_1707x775.jpg', cv2.IMREAD_UNCHANGED)
    fragment_positions = []

    for i in range(0, 327):
        print(f"{(i / 327) * 100}%")
        x, y, angle = -1, -1, -1
        fragment = cv2.imread(f'Michelangelo/frag_eroded/frag_eroded_{i}.png', cv2.IMREAD_UNCHANGED)
        x, y, angle = get_matches(fresque, fragment)

        if x >= 0:
            fragment_positions.append((i, x, y, angle))

    write_file(fragment_positions, "solution_euclide.txt")

if __name__ == "__main__":
    main()
