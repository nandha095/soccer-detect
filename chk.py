from ultralytics import YOLO
import cv2


model = YOLO("models/best.pt")

# # Perform prediction
# results = model.predict(
#     source=1,
#     conf=0.2,
#     show=True,
#      save=True,
#      save_dir="output_directory"
# )
print(model.names)
