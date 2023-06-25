import cv2
import face_recognition
import os

# Option 1: Take images and their names
def take_images():
    names = []
    images = []

    num_images = int(input("Enter the number of images you want to capture: "))

    for i in range(num_images):
        name = input("Enter the name of person {}: ".format(i+1))
        names.append(name)

        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("Unable to open the webcam")
            exit()

        while True:
            ret, frame = cap.read()
            cv2.imshow('Capture Images', frame)

            if cv2.waitKey(1) == ord('p'):  # Press 'p' to capture the image
                images.append(frame)
                break
            elif cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
                break

        cap.release()
        cv2.destroyAllWindows()

    # Save the captured images
    folder_path = "person/images"  # Replace with the desired folder path
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    for i, image in enumerate(images):
        file_name = "{}.jpg".format(names[i])  # Save image with the entered name
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, image)

     # Show the option for detecting persons
    print("Images captured successfully!")
    detect_person()

    # Return the captured images and their names
    return images, names

# Option 2: Detect the person
def detect_person():
     # Read the known images and names from the "person/images" folder
    folder_path = "person/images"  # Replace with the correct folder path
    known_images = []
    known_names = []

    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg"):
            # Append the image file path to known_images list
            image_path = os.path.join(folder_path, file_name)
            known_images.append(image_path)

            # Extract the name from the file name (remove the extension)
            name = os.path.splitext(file_name)[0]
            known_names.append(name)

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

# Main menu
def main_menu():
    print("Options:")
    print("1. Take images and their names")
    print("2. Detect the person")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        take_images()
    elif choice == "2":
        detect_person()
    else:
        print("Invalid choice. Please try again.")
        main_menu()

# Start the program
main_menu()
