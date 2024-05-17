from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter

# Load the YOLOv8 model
model= YOLO('/home/aims/obb_contents/weights/v8/best_04_10_ nano.pt') 

# Open the video file
cap = cv2.VideoCapture("/home/aims/obb_contents/rock/test_rock1.avi")

# cap = cv2.VideoCapture("/home/aims/2024/dataset/video/output_9.avi")
# Store the track history
track_history = defaultdict(lambda: [])
heatmap = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.float32)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        heatmap = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), dtype=np.float32)
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot() 

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        if results[0].boxes.id == None:
            cv2.imshow("YOLOv8 Tracking with Heatmap", annotated_frame)
            continue
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls = results[0].boxes.cls
    
        # Visualize the results on the frame

        # Plot the tracks
        for box, track_id, cl in zip(boxes, track_ids,cls):
            if cl == 0 : continue
            x, y, w, h = box
            track = track_history[track_id]
            track.append((int(x), int(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)
            # Update heatmap from the tracks
            for (x, y) in track:
                heatmap[y:y+2, x:x+2] += 1  # Simple increment
        # Apply Gaussian smoothing to the heatmap
        smoothed_heatmap = gaussian_filter(heatmap, sigma=10)
        heatmap_img = np.uint8(255 * smoothed_heatmap / np.max(smoothed_heatmap))

        # Convert heatmap to color
        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

        # Overlay the heatmap on the annotated frame
        cv2.addWeighted(heatmap_color, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        # Display the annotated frame with heatmap
        cv2.imshow("YOLOv8 Tracking with Heatmap", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()