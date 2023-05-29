import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

class HandSignRecognizer:
    def __init__(self, model_path):
        # Load the hand sign recognition model
        self.model = load_model(model_path)

        # Initialize the hand detection module
        self.mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)

        # Initialize the OpenCV video stream
        self.vs = cv2.VideoCapture(0)

        # ...

    def find_hands(self, image, draw=True, flip_type=False):
        # Convert the image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use the mediapipe hands module for hand detection
        results = self.mp_hands.process(image)

        hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = y_min = np.inf
                x_max = y_max = -np.inf
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y

                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                hands.append({'bbox': bbox})

                if draw:
                    # Draw the bounding box on the image
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        if flip_type:
            image = cv2.flip(image, 1)

        if draw:
            cv2.imshow("Hand Detection", image)

        return hands

    def preprocess_image(self, image):
        # Resize the image to the desired input shape for the model
        image = cv2.resize(image, (64, 64))

        # Normalize the pixel values to the range [0, 1]
        image = image / 255.0

        # Expand the dimensions to match the model's input shape (batch size = 1)
        image = np.expand_dims(image, axis=0)

        return image

    def predict_hand_sign(self, image):
        # Resize the preprocessed image
        resized_image = cv2.resize(image, (400, 400))

        # Expand dimensions to match the input shape of the model
        resized_image = np.expand_dims(resized_image, axis=0)

        # Perform prediction using the model
        predictions = self.model.predict(resized_image)

        # Get the predicted hand sign
        predicted_class_index = np.argmax(predictions[0])
        predicted_hand_sign = self.class_labels[predicted_class_index]

        return predicted_hand_sign

    def run(self):
        while True:
            # Read frame from video stream
            ret, frame = self.vs.read()

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Find hands in the frame
            hands = self.find_hands(frame, draw=False, flip_type=True)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                image = frame[y - 10:y + h + 10, x - 10:x + w + 10]

                # Perform hand sign prediction
                predicted_hand_sign = self.predict_hand_sign(image)

                # Display the predicted hand sign on the frame
                cv2.putText(frame, predicted_hand_sign, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Hand Sign Recognition", frame)

            # Check for the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video stream and close all windows
        self.vs.release()
        cv2.destroyAllWindows()


# Create an instance of the HandSignRecognizer class and run the program
model_path = './cnn8grps_rad1_model.h5'  # Replace with the path to your hand sign recognition model
hand_sign_recognizer = HandSignRecognizer(model_path)
hand_sign_recognizer.run()
