from __future__ import print_function
import sys
import numpy as np
import cv2
from imutils.video import FileVideoStream, VideoStream 
import openpyxl
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
        # Create a tracker based on tracker name
        if trackerType == trackerTypes[0]:
            tracker = cv2.legacy.TrackerBoosting_create()
        elif trackerType == trackerTypes[1]:
            tracker = cv2.legacy.TrackerMIL_create()
        elif trackerType == trackerTypes[2]:
            tracker = cv2.legacy.TrackerKCF_create()
        elif trackerType == trackerTypes[3]:
            tracker = cv2.legacy.TrackerTLD_create()
        elif trackerType == trackerTypes[4]:
            tracker = cv2.legacy.TrackerMedianFlow_create()
        elif trackerType == trackerTypes[5]:
            tracker = cv2.legacy.TrackerGOTURN_create()
        elif trackerType == trackerTypes[6]:
            tracker = cv2.TrackerMOSSE_create()
        elif trackerType == trackerTypes[7]:
            tracker = cv2.legacy.TrackerCSRT_create()
        else:
            tracker = None
            print('Incorrect tracker name')
            print('Available trackers are:')
            for t in trackerTypes:
                print(t)

        return tracker

# initialize the video capture object
cap = cv2.VideoCapture("test2.mp4")

# grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)

## Select boxes
bboxes = []
colors = [] 

frame_counter = 0

# Create a new Excel workbook and select the active worksheet
wb = openpyxl.Workbook()
ws = wb.active

# Write headers to the Excel sheet
ws.append(["Frame", "Contours", "X", "Y", "Width", "Height"])

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow('MultiTracker', cv2.WINDOW_NORMAL)

while True:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner
  bbox = cv2.selectROI('MultiTracker', frame)
  cv2.resizeWindow('MultiTracker', 720, 480)
  bboxes.append(bbox)
  #colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
  colors.append((255, 255, 255))
  print("Press q to quit selecting boxes and start tracking")
  print("Press any other key to select next object")
  k = cv2.waitKey(0) & 0xFF
  print(k)
  if (k == 113):  # q is pressed
    break

# Specify the tracker type
trackerType = "CSRT"
createTrackerByName(trackerType)

# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()

# Initialize MultiTracker 
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("There is no more frame to read, exiting...")
        break

    frame_counter += 1
    
    # get updated location of objects in subsequent frames
    ret, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # convert from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 # lower and upper limits for the red color
    lower_limit = np.array([131, 138, 171])
    upper_limit = np.array([179, 255, 244])

    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)

    mask = cv2.dilate(mask, np.ones((5,5)), iterations=2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Initialize label counter
    label_counter = 0

    # Iterate through contours
    for contour in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Increment label counter
        label_counter += 1
        
        ws.append([frame_counter, f"Contour {label_counter}", x, y, w, h])
       
        # Write label next to the contour
        cv2.putText(frame, f"Contour {label_counter}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    for bbox in boxes:
        bx = bbox[0]
        by = bbox[1]
        bw = bbox[2]
        bh = bbox[3]

        label_counter += 1

        ws.append([frame_counter, f"Contour {label_counter}", bx, by, bw, bh])
        
        #cv2.putText(frame, f"Contour {label_counter}", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    cv2.resizeWindow('frame', 1280, 720)
    # write the frame to the output file
    output.write(frame)
    if cv2.waitKey(30) == ord('q'):
        break

# Save the Excel workbook
wb.save("AOI_data.xlsx")

cap.release()
output.release()
cv2.destroyAllWindows()