import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from typing import List
from threading import Thread
import argparse

# FastAPI App initialization
app = FastAPI()

# YOLO model initialization
model = YOLO('yolov8n.pt')
names = model.names

# Global variables
drawing = False
points = []  # Global list to store points for the polygon
tracked_people = {}  # Dictionary to track entry/exit status of each person
entry_count = 0  # Track people entering the polygon
exit_count = 0  # Track people exiting the polygon
history = []  # To keep track of history
live_count = 0  # Live count of people inside the polygon

# Pydantic models for API configuration
class AreaConfig(BaseModel):
    points: List[List[int]]  # List of points for the polygon

# Mouse callback function for drawing the polygon
def mouse_drawing(event, x, y, flags, param):
    global points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click to add points
        if drawing and len(points) < 4:
            points.append((x, y))  # Add point to the list
        if len(points) == 4:  # Once 4 points are added, close the polygon
            drawing = False
        print(f"Point added: {x}, {y}")
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to reset the polygon
        points.clear()
        print("Polygon reset.")
        drawing = True

# Video processing function for people tracking and polygon detection
def process_video(url):
    global points, tracked_people, entry_count, exit_count, history, live_count
    cap = cv2.VideoCapture(url)
    cv2.namedWindow("YOLOv8 Tracking")
    cv2.setMouseCallback("YOLOv8 Tracking", mouse_drawing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of Video")
            break

        # Run YOLO detection
        results = model.track(frame, tracker="bytetrack.yaml", persist=True)
        # Draw polyline if points are selected
        if len(points) == 4:  # Once 4 points are selected, draw the polyline
            poly_points = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [poly_points], isClosed=True, color=(0, 255, 0), thickness=2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
            confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                c = names[class_id]
                if 'person' in c:  # We are interested in detecting people
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Check if the center of the bounding box is inside the polygon
                    is_inside = -1  # Default value to ensure it always gets assigned
                    if len(points) == 4:
                        # Check if the point (center of the box) is inside the polygon
                        is_inside = cv2.pointPolygonTest(np.array(points, np.int32), center, False)

                    # Track entry and exit of each person
                    if is_inside >= 0 and track_id not in tracked_people:
                        tracked_people[track_id] = {'entered': True, 'exited': False}  # Mark as entered
                        entry_count += 1
                        live_count += 1
                        history.append({
                            "track_id": track_id,
                            "event": "entered",
                            "timestamp": str(datetime.now()),
                            "coordinates": points  # Add the coordinates when the person enters
                        })
                        print(f"Person {track_id} entered.")
                    elif is_inside < 0 and track_id in tracked_people and not tracked_people[track_id]['exited']:
                        tracked_people[track_id]['exited'] = True  # Mark as exited
                        exit_count += 1
                        live_count -= 1
                        history.append({
                            "track_id": track_id,
                            "event": "exited",
                            "timestamp": str(datetime.now()),
                            "coordinates": points  # Add the coordinates when the person exits
                        })
                        print(f"Person {track_id} exited.")

                    # Draw the bounding box for the people detected inside the polygon
                    if is_inside >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
                        cv2.putText(frame, f'{track_id}: person', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Display entry and exit count
        cv2.putText(frame, f'Entry Count: {entry_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Exit Count: {exit_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("YOLOv8 Tracking", frame)

        key = cv2.waitKey(25)

        if cv2.waitKey(1) & key == ord("q"):
            break
        elif key == ord('p'):  # Pause the video and allow point selection
            print("Paused, select points on the video frame.")
            cv2.putText(frame, 'Paused, select points on the video frame.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.waitKey(-1)  # Wait for a key press to resume
        elif key == ord('r'):  # Reset points and allow re-drawing the polygon
            print("Resetting points, please select a new polygon.")
            points = []  # Clear points list to start a new polygon

# FastAPI endpoints for tracking and configuration

@app.get("/api/stats/")
async def get_history():
    """
    Endpoint to get the history of people coming in and out of the region
    """
    return history

@app.get("/api/stats/live")
async def get_live_count():
    """
    Endpoint to get the current number of people inside the region
    """
    return {"live_count": live_count}

@app.post("/api/config/area")
async def set_area(config: AreaConfig):
    """
    Endpoint to update the polygon region by providing new coordinates.
    """
    global points
    points = config.points
    return {"message": "Area configuration updated successfully", "new_area": points}

# Run FastAPI and OpenCV in separate threads
def start_api():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 tracking application.")
    parser.add_argument('--url', type=str, required=True, help="URL of the video stream.")
    return parser.parse_args()

if __name__ == "__main__":

    # Parse arguments
    args = parse_args()
    # Start FastAPI in a separate thread
    api_thread = Thread(target=start_api)
    api_thread.start()

    # Start processing the video
    process_video(args.url)
