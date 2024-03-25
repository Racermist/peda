import cv2

# Load pre-trained pedestrian classifier
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Load the video
video_capture = cv2.VideoCapture('C:/Users/astha/Downloads/PRO-106-ProjectTemplate-main/PRO-106-ProjectTemplate-main/walking.avi')

while True:
    # Read the frame
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect pedestrians in the frame
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Pedestrian Detection', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
