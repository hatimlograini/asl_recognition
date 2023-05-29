import traceback
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk


class SignLanguageDetection:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection")
        self.root.geometry("800x600")
        self.panel = tk.Label(root)
        self.panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.vs = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            cv2image = cv2.flip(frame, 1)
            cv2image_copy = np.array(cv2image)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            results = self.hands.process(cv2image_copy)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(cv2image_copy, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            self.root.after(10, self.video_loop)
        except Exception as e:
            traceback.print_exc()
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    SignLanguageDetection(root)
    root.mainloop()
