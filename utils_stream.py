import cv2


def frame_reader(path, stride=1, resize_to=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                if resize_to is not None:
                    frame = cv2.resize(frame, resize_to)
                yield frame
            idx += 1
    finally:
        cap.release()


def video_writer(path, fps, w, h, is_color=True, fourcc_str="XVID"):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h), is_color)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {path}")
    return writer
