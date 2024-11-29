import cv2
import dlib

# Frontal face detector
detector = dlib.get_frontal_face_detector()
# print type of detector
print(type(detector))
# Predictor for 5 face landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get image from camera
cap = cv2.VideoCapture(0)

while True:
    # Read image
    ret, frame = cap.read()
    # Check if image is not empty
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Detect faces
    faces = detector(gray)

    # Loop through faces
    for face in faces:
        top = face.top()
        bottom = face.bottom()
        left = face.left()
        right = face.right()

        # Draw rectangle 
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        # landmarks
        landmarks = predictor(gray, face)
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            frame = cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        # Show image
        cv2.imshow("frame", frame)
        
        # Crop face
        face_img = frame[top:bottom, left:right]
        face_img = cv2.resize(face_img, (224, 224))
        
        # Show face
        cv2.imshow("face_img_landmark", face_img)
    # Wait for key press
    key = cv2.waitKey(1)
    # If key 'q' is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()


  

    