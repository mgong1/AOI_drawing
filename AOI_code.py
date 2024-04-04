import numpy as np
import cv2
from imutils.video import FileVideoStream, VideoStream 
import openpyxl

# initialize the video capture object
cap = cv2.VideoCapture("test.mp4")

# grab the width, height, and fps of the frames in the video stream.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# initialize the FourCC and a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

frame_counter = 0

# Create a new Excel workbook and select the active worksheet
wb = openpyxl.Workbook()
ws = wb.active

# Write headers to the Excel sheet
ws.append(["Frame", "X", "Y", "Width", "Height"])

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("There is no more frame to read, exiting...")
        break

    frame_counter += 1

    # convert from BGR to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 # lower and upper limits for the red color
    lower_limit = np.array([0, 135, 211])
    upper_limit = np.array([179, 255, 255])

    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
#    cv2.imshow('mask', mask)
#    bbox = cv2.boundingRect(mask)
    mask = cv2.dilate(mask, np.ones((5,5)), iterations=2)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Iterate through contours
    for contour in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        ws.append([frame_counter, x, y, w, h])

    # if we get a bounding box, use it to draw a rectangle on the image
    # if bbox is not None:
    #     x, y, w, h = bbox
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # else:
    #     print("Object not detected")

    cv2.imshow('frame', frame)
    # write the frame to the output file
    output.write(frame)
    if cv2.waitKey(30) == ord('q'):
        break

# Save the Excel workbook
wb.save("contours_data.xlsx")

cap.release()
output.release()
cv2.destroyAllWindows()