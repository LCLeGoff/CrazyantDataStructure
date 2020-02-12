import cv2
import numpy as np


class Movies:
    def __init__(self, add, id_exp):
        self.add = add
        self.cap = cv2.VideoCapture(self.add)
        self.id_exp = id_exp

    def display(self):

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self, frame, grayscale=True):
        self.cap = cv2.VideoCapture(self.add)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame_img = self.cap.read()
        if grayscale is True:
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        return frame_img

    def get_next_frame(self, grayscale=True):
        try:
            ret, frame_img = self.cap.read()
            if grayscale is True:
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
            cv2.destroyAllWindows()
            return frame_img
        except cv2.error:
            return np.zeros((1080, 1920))
