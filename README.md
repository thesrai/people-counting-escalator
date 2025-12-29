# People Counting on Escalator

A computer vision project for counting people moving up and down on escalator video
using YOLOv8 for detection and SORT for tracking.

---

## Overview
This project detects and tracks people in escalator videos and counts them
based on their movement direction (up or down).

The main goal is to avoid double counting by assigning a unique ID to each person
and counting them only once when they cross predefined virtual lines.

---

## Features
- People detection using YOLOv8
- Object tracking with SORT algorithm
- Direction-based counting (Up / Down)
- ROI mask and limit lines determined using a mouse-click script
- Prevention of double counting using unique track IDs

---

## Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- SORT (Simple Online and Realtime Tracking)
- NumPy
- SciPy
- CvZone
- torch

---

## How It Works
1. The input video is read frame by frame.
2. A Region of Interest (ROI) mask is applied to limit detection to the escalator area.
3. YOLOv8 detects people in each frame.
4. Detected bounding boxes are passed to the SORT tracker.
5. Each person receives a unique ID.
6. When the center of a tracked person crosses a predefined line:
   - The ID is added to a set.
   - The person is counted only once.
7. The total number of people moving up and down is displayed on the video.

---

## Limitation
Children walking close to adults may not be detected due to occlusion and small object size.

---

## Installation
Clone the repository and install the required dependencies.

---

## Usage
Use the video in the assets folder.
Run the script:  python peopleCounter.py

---

## License
This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

You are free to:

Use the code

Modify the code

Distribute the code

Under the condition that the same license is preserved.

See the LICENSE file for more details.

---

## Acknowledgments
This project uses the SORT (Simple Online and Realtime Tracking) algorithm
developed by Alex Bewley.

Original repository:
https://github.com/abewley/sort

Original implementation:
Copyright (C) 2016–2020 Alex Bewley

Licensed under the GNU General Public License v3.0 (GPL-3.0).

---

## Third-Party Code and Modifications

This project includes a modified version of the SORT (Simple Online and Realtime Tracking)
algorithm originally developed by Alex Bewley and released under the GPL-3.0 license.

The original code was modified to simplify the linear assignment logic and remove
unused dependencies that caused compatibility issues in this project.
