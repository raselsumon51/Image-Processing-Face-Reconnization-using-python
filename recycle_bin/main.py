import cv2
import face_recognition

# Load the known images and their corresponding names
known_images = ["person1.jpg", "rasel.jpg","muzahid.jpg"]  # Add the paths or filenames of known person images
known_names = ["John", "Rasel","Muzahid"]  # Add the corresponding names of the known persons

# Load the images and encode the known faces
known_faces = []
for image_path in known_images:
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)

# Load the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Unable to open the webcam")
    exit()

# Process frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to speed up face detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Initialize an empty list for names
    face_names = []

    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        # If a match is found, get the name of the known person
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Add the name to the list of names
        face_names.append(name)

    # Display the name for each detected face
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up the face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (0, 0, 0), 1)

    # Display the resulting frame
    cv2.imshow('Person Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
