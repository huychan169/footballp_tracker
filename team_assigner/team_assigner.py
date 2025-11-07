import cv2
import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import KMeans

def _clip_bbox(frame, bbox):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)

def _center_torso_crop(img, ratio=(0.2, 0.7, 0.2, 0.8)):
    h, w = img.shape[:2]
    y_top = max(0, int(h * ratio[0]))
    y_bot = min(h, max(y_top + 1, int(h * ratio[1])))
    x_l   = max(0, int(w * ratio[2]))
    x_r   = min(w, max(x_l + 1, int(w * ratio[3])))
    return img[y_top:y_bot, x_l:x_r]

def _remove_grass_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([25,  30,  30], dtype=np.uint8)  
    upper1 = np.array([95, 255, 255], dtype=np.uint8)  
    grass  = cv2.inRange(hsv, lower1, upper1)
    mask   = cv2.bitwise_not(grass)

    kernel = np.ones((3,3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask  # 255 = giữ (không phải cỏ)

def _hsv_hist_features(img_bgr, mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    histH = cv2.calcHist([hsv],[0],mask,[24],[0,180]).flatten()
    histS = cv2.calcHist([hsv],[1],mask,[16],[0,256]).flatten()
    histV = cv2.calcHist([hsv],[2],mask,[8 ],[0,256]).flatten()
    feat  = np.concatenate([histH, histS, histV]).astype(np.float32)
    n = np.linalg.norm(feat) + 1e-6
    return feat / n

def _lab_stats(img_bgr, mask):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:,:,1][mask>0].astype(np.float32)
    b = lab[:,:,2][mask>0].astype(np.float32)
    if a.size < 10:  # bảo vệ
        return np.zeros(4, np.float32)
    return np.array([a.mean(), a.std(), b.mean(), b.std()], dtype=np.float32)

def _dominant_color_non_grass(img_bgr, mask, k=2):
    pts = img_bgr[mask>0].reshape(-1,3)
    if pts.shape[0] < 50:
        return np.zeros(3, np.float32)
    Z = np.float32(pts)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    dom = centers[np.argmax(counts)]
    return dom.astype(np.float32)  # BGR

def extract_jersey_features(frame, bbox):
    clipped = _clip_bbox(frame, bbox)
    if clipped is None:
        return None
    x1,y1,x2,y2 = clipped
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0 or min(patch.shape[:2]) < 5:
        return None

    torso = _center_torso_crop(patch)
    if torso.size == 0:
        return None

    mask_keep = _remove_grass_hsv(torso)
    # mask quá ít điểm -> fallback giữ 
    if np.count_nonzero(mask_keep) < 50:
        mask_keep = np.ones(torso.shape[:2], np.uint8) * 255

    f_hist = _hsv_hist_features(torso, mask_keep)
    f_lab  = _lab_stats(torso, mask_keep)
    f_dom  = _dominant_color_non_grass(torso, mask_keep) / 255.0  

    return np.concatenate([f_hist, f_lab, f_dom]).astype(np.float32)  

class TeamAssigner:
    def __init__(self, flip_guard_frames=8):
        self.team_centers = {}         
        self.player_team_dict = {}     
        self.kmeans = None
        self.history = defaultdict(lambda: deque(maxlen=flip_guard_frames)) 
        self.flip_guard_frames = flip_guard_frames

    def assign_team_model(self, frame, player_detections):
        feats = []
        for _, det in player_detections.items():
            bbox = det.get("bbox")
            if bbox is None:
                continue
            feat = extract_jersey_features(frame, bbox)
            if feat is not None:
                feats.append(feat)

        if len(feats) < 2:
            self.kmeans = None
            self.team_centers.clear()
            return

        X = np.asarray(feats, dtype=np.float32)
        km = KMeans(n_clusters=2, init="k-means++", n_init="auto", random_state=42)
        km.fit(X)
        self.kmeans = km
        self.team_centers[1] = km.cluster_centers_[0]
        self.team_centers[2] = km.cluster_centers_[1]

    def _predict_team_once(self, frame, bbox):
        if self.kmeans is None:
            return None
        feat = extract_jersey_features(frame, bbox)
        if feat is None:
            return None
        label = int(self.kmeans.predict(feat.reshape(1,-1))[0]) + 1  # {1,2}
        return label

    def get_player_team(self, frame, player_bbox, player_id):
        label = self._predict_team_once(frame, player_bbox)
        if label is None:
            return self.player_team_dict.get(player_id, None)

        # chống nhấp nháy
        h = self.history[player_id]
        h.append(label)

        votes1 = sum(1 for v in h if v == 1)
        votes2 = len(h) - votes1
        stable = 1 if votes1 > votes2 else 2

        # flip nếu đủ trong queue
        if len(h) == h.maxlen:
            self.player_team_dict[player_id] = stable
        else:
            self.player_team_dict.setdefault(player_id, stable)

        return self.player_team_dict[player_id]

    @property
    def team_colors(self):
        colors = {}
        if self.kmeans is not None:
            for i in [1,2]:
                ctr = self.team_centers.get(i, None)
                if ctr is not None and ctr.shape[0] >= 3:
                    bgr = (ctr[-3:]*255.0).clip(0,255).astype(np.uint8).tolist()
                    colors[i] = tuple(int(c) for c in bgr)
        return colors
