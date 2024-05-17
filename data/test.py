import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLOv8 model
model = YOLO('/home/aims/obb_contents/weights/v8/best_04_10_ nano.pt') 
# Open the video file
cap = cv2.VideoCapture("/home/aims/2024/dataset/video/output_9.avi")
# Store the track history
track_history = defaultdict(list)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Reset heatmap for each frame
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id == None:
            continue
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Update the heatmap for each track
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            cx, cy = int(x + w / 2), int(y + h / 2)  # Center point of the box
            track_history[track_id].append((cx, cy))  # Append new center point to the track history

        # Generate heatmap based on tracks
        for track in track_history.values():
            for x, y in track:
                heatmap[y:y+2, x:x+2] += 1  # Simple increment to simulate a point

        # Normalize the heatmap to use as an alpha channel
        heatmap_normalized = np.uint8(255 * heatmap / np.max(heatmap))
        alpha_channel = gaussian_filter(heatmap_normalized, sigma=10)  # Smooth the alpha channel

        # Create an RGBA image by combining with the alpha channel
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        rgba_frame[:, :, 3] = alpha_channel

        # Extract RGB and Alpha Channels
        rgb_frame = rgba_frame[:, :, :3]
        alpha_frame = rgba_frame[:, :, 3] / 255.0

        # Composite the RGB frame with the heatmap using the alpha channel
        composite_frame = (rgb_frame * alpha_frame[:, :, None]).astype(np.uint8)

        # Display the composite frame
        cv2.imshow('Composite Frame', composite_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()