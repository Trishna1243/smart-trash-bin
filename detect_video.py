import cv2
import torch

# Load trained model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='best.pt', force_reload=True)

cap = cv2.VideoCapture("sample.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results.render()[0]

    out.write(annotated)

cap.release()
out.release()

print("Done")