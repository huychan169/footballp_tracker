from typing import Any, Dict, List, Tuple, Optional
import cv2
import numpy as np
from numpy import ndarray

from src.datatools.reader import read_annot

def find_closest_points(points, x, y, any_side=False):
    distances = []
    for i, point in enumerate(points):
        distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)
        distances.append((distance, i))

    distances.sort(key=lambda p: p[0])

    idx1 = distances[0][1]
    x1 = points[idx1][0]
    y1 = points[idx1][1]

    if any_side:
        idx2 = distances[1][1]
        return np.vstack((points[idx1], points[idx2]))

    for i in range(1, len(distances)):
        idx2 = distances[i][1]
        x2 = points[idx2][0]
        y2 = points[idx2][1]
        if ((x1 <= x <= x2) or (x1 >= x >= x2)) and ((y1 <= y <= y2) or (y1 >= y >= y2)):
            return np.vstack((points[idx1], points[idx2]))
    return None


LINE_CLS: Dict[int, str] = {
    0: 'Goal left post left ',
    1: 'Goal right post right',
    2: 'Middle line',
    3: 'Small rect. right top',
    4: 'Side line bottom',
    5: 'Goal right post left',
    6: 'Big rect. right main',
    7: 'Goal left crossbar',
    8: 'Small rect. left bottom',
    9: 'Side line left',
    10: 'Big rect. right top',
    11: 'Small rect. left top',
    12: 'Side line right',
    13: 'Big rect. left top',
    14: 'Goal left post right',
    15: 'Small rect. right bottom',
    16: 'Side line top',
    17: 'Goal right crossbar',
    18: 'Small rect. left main',
    19: 'Big rect. left main',
    20: 'Big rect. right bottom',
    21: 'Small rect. right main',
    22: 'Big rect. left bottom'
}


def map_point2pixel(
    line1: Tuple[float, float],
    line2: Tuple[float, float],
    img_size: Tuple[int, int] = (960, 540)
) -> Tuple[np.ndarray, np.ndarray]:

    line1_arr = np.array(line1) * img_size
    line2_arr = np.array(line2) * img_size
    return line1_arr, line2_arr


def get_extreme_points(
    points_data: Dict[str, List[Any]],
    img_size: Tuple[int, int] = (960, 540)
) -> Dict[int, Optional[Tuple[Tuple[ndarray, ndarray], Any]]]:

    res: Dict[int, Optional[Tuple[Tuple[ndarray, ndarray], Any]]] = {}

    for i, line_info in LINE_CLS.items():
        res[i] = None
        if line_info in points_data:
            points = points_data[line_info][0]
            line_param = points_data[line_info][1]
            if len(points) > 1:
                points_extreme = map_point2pixel(points[0], points[-1], img_size)
                res[i] = (points_extreme, line_param)

    return res


def sort_anno(
    annos: Dict[str, List[Tuple[float, float]]],
    img_size: Tuple[int, int] = (960, 540)
) -> Tuple[Dict[str, Any], bool]:

    annos_update: Dict[str, Any] = {}
    usable_flag = True

    for k, points in annos.items():
        if k in ('Circle left', 'Circle right', 'Circle central', 'Line unknown'):
            continue
        else:
            if len(points) >= 2:
                points, slope, intercept = sort_points_on_line(points, input_size=img_size)
                if slope is not None and intercept is not None:
                    annos_update[k] = (points, (slope, intercept))
                else:
                    usable_flag = False
            else:
                usable_flag = False

    return annos_update, usable_flag


def filter_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    filtered_pairs = [(x_i, y_i) for x_i, y_i in zip(x, y)
                      if np.isfinite(x_i) and np.isfinite(y_i)]

    if len(filtered_pairs) == 0:
        return np.array([]), np.array([])

    x_filtered, y_filtered = zip(*filtered_pairs)
    return np.array(x_filtered), np.array(y_filtered)


def sort_points_on_line(
    points: List[Tuple[float, float]],
    input_size: Tuple[int, int]
) -> Tuple[List[Any], Any, Any]:

    w, h = input_size
    x = np.array([p[0] for p in points]) * w
    y = np.array([p[1] for p in points]) * h

    x, y = filter_xy(x, y)

    if x.size >= 2 and y.size >= 2 and np.std(x) != 0 and np.std(y) != 0:
        slope, intercept = np.polyfit(x, y, deg=1)
    else:
        return points, None, None

    line_origin = np.array([0, intercept])
    line_direction = np.array([1, slope])
    line_direction_normalized = line_direction / (np.linalg.norm(line_direction) + 1e-5)

    projected_positions = []
    for p in points:
        point_vector = np.array(p) - line_origin
        projected_position = np.dot(point_vector, line_direction_normalized)
        projected_positions.append(projected_position)

    sorted_points = [p for _, p in sorted(zip(projected_positions, points))]

    return sorted_points, slope, intercept


def loc2img(
    img: np.ndarray,
    points: Dict[int, Optional[Tuple[Tuple[ndarray, ndarray], Any]]],
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:

    for ps in points.values():
        pixels = []
        if ps is not None:
            for p in ps[0]:
                pt = (round(p[0]), round(p[1]))
                cv2.circle(img, pt, radius=5, color=color, thickness=2)
                pixels.append(pt)

            if len(pixels) > 1:
                cv2.line(img, pixels[0], pixels[-1], color=color, thickness=1)

    return img


if __name__ == "__main__":
    anno_path = '/workdir/data/dataset/train/00000.json'
    img_path = '/workdir/data/dataset/train/00000.jpg'

    sample = read_annot(anno_path)
    ann = sort_anno(sample)
    extreme = get_extreme_points(ann[0])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_pred = loc2img(img, extreme)
    cv2.imshow('Draw lines', img_pred)
    cv2.waitKey()
