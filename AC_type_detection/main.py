

from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. Loading a model configuration and train the model from scratch
model = YOLO("yolov8n.yaml")  # Build a new model from scratch
results = model.train(data="config.yaml", epochs=1)  # Train the model with your custom dataset

# 2. Loading the trained model for inference
trained_model_path = "D:/Test_programs/ma_project/ObjectDetc/runs/detect/train/weights/best.pt"
model = YOLO(trained_model_path)  # Load the trained model for predictions

# 3. Run prediction on a single image
image_test = model.predict(source="D:/Test_programs/ma_project/ObjectDetc/test_images/sp.jpg", conf=0.25, save=True)

# 4. Display the result
# Check if any detections were made by printing the result information
if len(image_test) > 0 and image_test[0].boxes is not None:
    print("Detections:", image_test[0].boxes)  # Printing bounding boxes and confidence scores

# 5. Result plotting if any detections is there
plt.imshow(image_test[0].plot())
plt.axis("off")
plt.show()



