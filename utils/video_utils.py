# import cv2

# def read_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     return frames

# def save_video(output_video_frames, output_video_path):
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
#     for frame in output_video_frames:
#         out.write(frame)
#     out.release()

import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# def save_video(output_video_frames, output_video_path, fps=24):
#     """Ghi video trực tiếp từng frame để không dồn RAM."""
#     if not output_video_frames:
#         return

#     height, width = output_video_frames[0].shape[:2]
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     for frame in output_video_frames:
#         out.write(frame)
#         # giải phóng bộ nhớ của frame sau khi ghi
#         del frame  

#     out.release()

def save_video(frames_iter, output_video_path, fps=24):
    first_frame = next(iter(frames_iter))
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # ghi frame đầu
    out.write(first_frame)

    # ghi các frame còn lại
    for frame in frames_iter:
        out.write(frame)

    out.release()
