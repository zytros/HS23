import cv2
import numpy as np



# Open the video file
input_video_path = 'ex6_data/video1.avi'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object to save the rotated video
output_video_path = 'ex6_data/video1_rotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (height, width))

# Read and process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rotate the frame by 90 degrees
    rotated_frame = cv2.transpose(frame)
    rotated_frame = cv2.flip(rotated_frame, 1)  # 1 means horizontal flip, adjust as needed

    # Write the rotated frame to the output video
    out.write(rotated_frame)

# Release video capture and writer objects
cap.release()
out.release()

print("Video processing complete.")
