from __future__ import absolute_import

import cv2
import time
import sys
sys.path.append('.')
from siamfc import TrackerSiamFC1


class UIControl:
    def __init__(self):
        self.mode = 'init'  # init, select, track
        self.target_tl = (-1, -1)
        self.target_br = (-1, -1)
        self.new_init = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.mode == 'init':
            self.target_tl = (x, y)
            self.target_br = (x, y)
            self.mode = 'select'
        elif event == cv2.EVENT_MOUSEMOVE and self.mode == 'select':
            self.target_br = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN and self.mode == 'select':
            self.target_br = (x, y)
            self.mode = 'init'
            self.new_init = True

    def get_tl(self):
        return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

    def get_br(self):
        return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

    def get_bb(self):
        tl = self.get_tl()
        br = self.get_br()

        bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
        return bb


if __name__ == '__main__':
    engine_path = 'pretrained/siamfc_alexnet_e50_dynamic.engine'
    tracker = TrackerSiamFC1(engine_path=engine_path)

    display_name = 'Display1'
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cap = cv2.VideoCapture(0)
    while True:
        rval, frame = cap.read()
        if frame is None:
            time.sleep(0.1)
            continue
        break
    cv2.imshow(display_name, frame)
    cv2.waitKey(300)
    ui_control = UIControl()
    cv2.setMouseCallback(display_name, ui_control.mouse_callback)

    running = True
    while running:
        rval, frame = cap.read()
        frame_disp = frame.copy()

        if ui_control.new_init:
            ui_control.new_init = False
            box = ui_control.get_bb()
            tracker.init(frame, box)
            tracker.tracking = True

        if tracker.tracking:
            box = tracker.update(frame)

        # Draw box
        if ui_control.mode == 'select':
            cv2.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (0, 0, 255), 2)

        if tracker.tracking:
            cv2.rectangle(frame_disp, box, (255, 0, 0), 2)            


        cv2.imshow(display_name, frame_disp)
        key = cv2.waitKey(1)
        if key==27:
            running = False
    cv2.destroyAllWindows()
