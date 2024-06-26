from __future__ import print_function
import sys
import cv2
from random import randint
import openpyxl


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


# Set video to load
videoPath = "test_output.mp4"

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

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
ws.append(["Frame", "X", "Y", "Width", "Height"])

# OpenCV's selectROI function doesn't work for selecting multiple objects in Python
# So we will call this function in a loop till we are done selecting all objects
while True:
  # draw bounding boxes over objects
  # selectROI's default behaviour is to draw box starting from the center
  # when fromCenter is set to false, you can draw box starting from top left corner
  bbox = cv2.selectROI('MultiTracker', frame)
  bboxes.append(bbox)
  colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
  print("Press q to quit selecting boxes and start tracking")
  print("Press any other key to select next object")
  k = cv2.waitKey(0) & 0xFF
  print(k)
  if (k == 113):  # q is pressed
    break

# print('Selected bounding boxes {}'.format(bboxes))
# print('(x,y):',bbox[0],bbox[1])

# Specify the tracker type
trackerType = "CSRT"
createTrackerByName(trackerType)

# Create MultiTracker object
multiTracker = cv2.legacy.MultiTracker_create()


# Initialize MultiTracker 
for bbox in bboxes:
  multiTracker.add(createTrackerByName(trackerType), frame, bbox)

  # Process video and track objects
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break
  
  frame_counter += 1

  # get updated location of objects in subsequent frames
  success, boxes = multiTracker.update(frame)

  # draw tracked objects
  for i, newbox in enumerate(boxes):
    p1 = (int(newbox[0]), int(newbox[1]))
    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

  for bbox in boxes:
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    ws.append([frame_counter, x, y, w, h])

  # show frame
  cv2.imshow('MultiTracker', frame)

  # quit on ESC button
  if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    break

# Save the Excel workbook
wb.save("balls_data.xlsx")

cap.release()
cv2.destroyAllWindows()