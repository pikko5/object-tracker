import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from typing import List
from threading import Thread
import argparse
import streamlit as st
import asyncio
from PIL import ImageColor
import torch
import os
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []

# FastAPI App initialization (will run in the background)
app = FastAPI()

# YOLO model initialization
model = YOLO('car_dent.pt')
names = model.names

# Global variables
tracked_people = {}  # Dictionary to track entry/exit status of each person
entry_count = 0  # Track people entering the frame
exit_count = 0  # Track people exiting the frame
history = []  # To keep track of history
live_count = 0  # Live count of people currently in the frame
detection_color = (0, 255, 128) # Default detection box color
text_color = (255, 0, 0) # Default text color

# Pydantic models for API configuration
class AreaConfig(BaseModel):
    points: List[List[int]]  # List of points for the polygon (will not be used)

@app.get("/api/stats/")
async def get_history():
    """
    Endpoint to get the history of people entering and exiting the frame
    """
    return history

@app.get("/api/stats/live")
async def get_live_count():
    """
    Endpoint to get the current number of people in the frame
    """
    return {"live_count": live_count}

@app.post("/api/config/area")
async def set_area(config: AreaConfig):
    """
    Endpoint to update the polygon region (no longer used).
    """
    return {"message": "Area configuration endpoint is no longer active."}

# Function to process video frames
def process_frame(frame):
    global tracked_people, entry_count, exit_count, history, live_count, detection_color, text_color

    # Run YOLO detection
    results = model.track(frame, tracker="bytetrack.yaml", persist=True)
    annotated_frame = frame.copy()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        current_tracked_ids = set(track_ids)

        # Handle exits: If a tracked ID is no longer detected
        exited_ids = set(tracked_people.keys()) - current_tracked_ids
        for track_id in exited_ids:
            if not tracked_people[track_id].get('exited', False):
                tracked_people[track_id]['exited'] = True
                exit_count += 1
                live_count -= 1
                history.append({
                    "track_id": track_id,
                    "event": "exited",
                    "timestamp": str(datetime.now())
                })
                print(f"Person {track_id} exited the frame.")

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            if 'person' not in c:  # We are interested in detecting people
                x1, y1, x2, y2 = box

                # Handle entries: If a new track ID is detected
                if track_id not in tracked_people:
                    tracked_people[track_id] = {'entered': True, 'exited': False}
                    entry_count += 1
                    live_count += 1
                    history.append({
                        "track_id": track_id,
                        "event": "entered",
                        "timestamp": str(datetime.now())
                    })
                    print(f"Person {track_id} entered the frame.")

                # Draw the bounding box for the detected person
                cv2.rectangle(annotated_frame, (x1 + 100, y1 + 100), (x2 - 100, y2 - 100), detection_color, 2)
                cv2.putText(annotated_frame, f'{track_id}: {c}', (x1+ 100, y1+ 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        # Update tracked people: remove those that have exited and are no longer in current detections
        tracked_people = {k: v for k, v in tracked_people.items() if k in current_tracked_ids or not v.get('exited', False)}

    # Display entry and exit count on the frame
    cv2.putText(annotated_frame, f'Damage Count: {entry_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # cv2.putText(annotated_frame, f'Exit Count: {exit_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # cv2.putText(annotated_frame, f'Live Count: {live_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    return annotated_frame

async def video_stream(url, frame_placeholder):
    global tracked_people, entry_count, exit_count, history, live_count
    cap = cv2.VideoCapture(url)

    # Reset tracking variables when a new video starts
    tracked_people = {}
    entry_count = 0
    exit_count = 0
    history = []
    live_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of Video Stream")
            break

        processed_frame = process_frame(frame)
        frame_placeholder.image(processed_frame, channels="BGR")
        await asyncio.sleep(0.03) # Adjust delay as needed

    cap.release()

def color_picker():
    global detection_color, text_color
    st.sidebar.subheader("Color Configuration")
    detection_color_str = st.sidebar.color_picker("Detection Box Color")
    text_color_str = st.sidebar.color_picker("Text Color")

    detection_color = ImageColor.getrgb(detection_color_str)
    text_color = ImageColor.getrgb(text_color_str)

def main():
    st.title("Real-time Damage Tracker")

    color_picker()

    video_url = st.text_input("Enter Video URL", "rtsp://your_rtsp_stream") # Replace with your default stream if needed

    st.subheader("Video Feed")
    st.info("The application will now track all damages recorded on the feed.")

    frame_placeholder = st.empty()

    if st.button("Start Video"):
        if video_url:
            asyncio.run(video_stream(video_url, frame_placeholder))
        else:
            st.error("Please enter a video URL.")

    st.subheader("Current Statistics")
    st.write(f"Entry Count: {entry_count}")
    st.write(f"Exit Count: {exit_count}")
    st.write(f"Live Count: {live_count}")

    if st.checkbox("Show Tracking History"):
        st.subheader("Tracking History")
        st.json(history)

    # Removed set_polygon_points_manually()

if __name__ == "__main__":
    main()

# To run this Streamlit app, save it as a Python file (e.g., app.py)
# and then run it from your terminal using: streamlit run app.py

# The FastAPI part is still included but won't be directly used by the Streamlit interface in this setup.
# If you need the API endpoints for other purposes, you would run the FastAPI app separately.