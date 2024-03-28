import cv2
import mediapipe as mp
import os
import yaml

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

sign = input("Enter the Name of Sign: ")
hand_dir = os.path.join("Signs", sign)

create_directory(hand_dir)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

# Set the margin for the bounding box
margin = 20

# Initialize counters
right_hand_count = 0
left_hand_count = 0

# Display initial message for 2 seconds
display_time = 0
initial_message_displayed = False

while True:
    _, frame = cap.read()

    if not initial_message_displayed:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, "Show sign using Right Hand", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        display_time += 1

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract bounding box coordinates with margin
            bbox_min = [float('inf'), float('inf')]  # Initialize with large values
            bbox_max = [float('-inf'), float('-inf')]  # Initialize with small values

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                # Update bounding box coordinates
                bbox_min[0] = min(bbox_min[0], x)
                bbox_min[1] = min(bbox_min[1], y)
                bbox_max[0] = max(bbox_max[0], x)
                bbox_max[1] = max(bbox_max[1], y)

            # Add margin to the bounding box
            bbox_min[0] = max(0, bbox_min[0] - margin)
            bbox_min[1] = max(0, bbox_min[1] - margin)
            bbox_max[0] = min(frame.shape[1], bbox_max[0] + margin)
            bbox_max[1] = min(frame.shape[0], bbox_max[1] + margin)

            # Draw bounding box
            cv2.rectangle(frame, (int(bbox_min[0]), int(bbox_min[1])),
                          (int(bbox_max[0]), int(bbox_max[1])), (0, 255, 0), 2)

            # Capture images and save hand landmarks data in YAML file
            if right_hand_count < 100 and display_time > 2:
                # Capture images only within the bounding box
                roi = frame[int(bbox_min[1]):int(bbox_max[1]), int(bbox_min[0]):int(bbox_max[0])]
                right_hand_data = {'hand_landmarks': []}

                for landmark in hand_landmarks.landmark:
                    right_hand_data['hand_landmarks'].append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})

                with open(os.path.join(hand_dir, f"right_{right_hand_count}.yml"), 'w') as yaml_file:
                    yaml.dump(right_hand_data, yaml_file, default_flow_style=False)

                right_hand_count += 1
            elif right_hand_count >= 100 and left_hand_count < 100:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)  # Black background
                cv2.putText(frame, "Show sign using Left Hand", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Capture images for left hand and save hand landmarks data in YAML file
                if left_hand_count < 100:
                    # Capture images only within the bounding box
                    roi = frame[int(bbox_min[1]):int(bbox_max[1]), int(bbox_min[0]):int(bbox_max[0])]
                    left_hand_data = {'hand_landmarks': []}

                    for landmark in hand_landmarks.landmark:
                        left_hand_data['hand_landmarks'].append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})

                    with open(os.path.join(hand_dir, f"left_{left_hand_count}.yml"), 'w') as yaml_file:
                        yaml.dump(left_hand_data, yaml_file, default_flow_style=False)

                    left_hand_count += 1

            # Close the camera after capturing 100 images for both hands
            if right_hand_count >= 100 and left_hand_count >= 100:
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Display the result
    cv2.imshow("Hand Tracking", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows (just in case)
cap.release()
cv2.destroyAllWindows()
