import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose  an image to detect faces in
#img = cv2.imread('RDJ.png')
#img = cv2.imread('rdi.jpg')
# To capture video from webcam
#webcam = cv2.VideoCapture('clips.mp4')
webcam = cv2.VideoCapture(0)
# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscale_ima = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_ima)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # display that webcam and tilte with the faces spotted
    cv2.imshow('Clever Programmer Face Detector', frame)
    # Listen for a key press for 1 millisecond, then move on
    key = cv2.waitKey(1)
    # Stop if Q key is pressed(Q=81 and q=113)
    if key == 81 or key == 113:
        break
# Release the VideoCapture object
webcam.release()


"""
# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_ima)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(255),
                  randrange(255), randrange(255)), 5)

# print(face_coordinates)


# display that image and tilte with the faces
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()


print("Code Completed")

"""
