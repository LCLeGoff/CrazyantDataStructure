import cv2
import matplotlib.pyplot as plt


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

    def get_frame(self, frame):

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame_img = self.cap.read()
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        return frame_img
