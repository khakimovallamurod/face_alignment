import cv2
import dlib

# Frontal face detector
detector = dlib.get_frontal_face_detector()
# print type of detector
print(type(detector))
# Predictor for 5 face landmarks
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# Get image from camera
cap = cv2.VideoCapture(0)

while True:
    # Read image
    ret, frame = cap.read()
    # Check if image is not empty
    if not ret:
        break

    # Show image
    cv2.imshow("frame", frame)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   


    # Detect faces
    faces = detector(gray)

    # Loop through faces
    for face in faces:
        top = face.top()
        bottom = face.bottom()
        left = face.left()
        right = face.right()

        # Crop face
        face_img = frame[top:bottom, left:right]
        # Show face
        cv2.imshow("face", face_img)
    # Wait for key press
    key = cv2.waitKey(1)
    # If key 'q' is pressed
    if key == ord("q"):
        break
 
 
 

# # Read image
# img = cv2.imread("face.jpg")
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect faces
# faces = detector(gray)

# # Loop through faces
# for face in faces:
#     top = face.top()
#     bottom = face.bottom()
#     left = face.left()
#     right = face.right()

#     # Crop face
#     face_img = img[top:bottom, left:right]
#     # Show face
#     cv2.imshow("face", face_img)
#     # Wait for key press
#     cv2.waitKey(0)
# Close window
cv2.destroyAllWindows()


  