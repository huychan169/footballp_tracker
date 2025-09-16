from ultralytics import YOLO 

model = YOLO ('models/best_v5lu_ep30_cl.pt')

results = model.predict('input_videos/cut30s.mp4', conf=0.5, save=True)
print(results[0])
print('================================')
for box in results[0].boxes:
    print(box)
