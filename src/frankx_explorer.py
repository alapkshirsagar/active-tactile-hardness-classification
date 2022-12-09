from frankx import (Gripper, Robot, LinearMotion, Affine, LinearRelativeMotion, Kinematics, NullSpaceHandling,
                    JointMotion, MotionData, Reaction, Measure, StopMotion)
import numpy as np
import time
from gelsight import gsdevice
import cv2 as cv
from datetime import datetime
from matplotlib import pyplot as plt
from threading import Thread


class GridExplorer():
    def __init__(self):
        self.robot = Robot("172.168.0.2")
        self.robot.set_default_behavior()  
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(0.01)

        #self.cam = gsdevice.Camera(gsdevice.Finger.MINI, 0)
        #self.cam.connect()
        self.cv_cam = cv.VideoCapture(0)
        #self.res = (0, 0, 320, 240)
        #print('image size = ', self.cam.get_raw_image().shape)

        #self.neutral_joints = [0.6118366520437153, 0.15716609762216865, -0.25319002091814896, -2.394232290291543, 0.029041619664173065, 2.5748198744888313, 1.1364431301769282]    # Calibration 25.11.2022
        #self.neutral_joints = [0.3707151330437521, 0.16326812298140542, -0.028099087464621014, -2.3989490835251646, 0.00702887462245093, 2.5924458934995864, 1.1128487507518794]    # Calibration 28.11.2022
        self.neutral_joints = [0.37210778693148966, 0.28854662857557595, -0.02800002464495207, -2.39302845971585, 0.01651057972216312, 2.7108360546139134, 1.106010951997505]
        self.homing_motion = JointMotion(self.neutral_joints)

        self.grid_size_x = 4
        self.grid_size_y = 5 

        self.cell_size = 0.09       

        self.pos = np.array([0,0])
        self.home()
        

    def home(self):
        self.robot.move(self.homing_motion)
        self.pos = np.array([0,0])

    def goto(self, pos_):
        if not (pos_[0] < self.grid_size_x and pos_[1] < self.grid_size_y):
            return
        if np.linalg.norm(pos_ - self.pos) < 0.01:
            return

     
        rel = pos_ - self.pos
        motion = LinearRelativeMotion(Affine(rel[0]*self.cell_size, -rel[1]*self.cell_size, 0.0))
        self.robot.move(motion)
   

        self.pos = pos_

    def push(self, max_force, record = False, name = None, framerate = 30): # force should be positive even though negative z direction
        max_force = np.abs(max_force) # just to be safe
      
        
        return_q = self.robot.get_state().q
        return_motion = JointMotion(return_q)
        
        push_constraint = MotionData().with_reaction(Reaction(Measure.ForceZ < -max_force, StopMotion()))
        push_motion = LinearRelativeMotion(Affine(0.0, 0.0, -0.1))

        if record:
            recording = True
            stop_recording = lambda: not recording
            recorder_thread = Thread(target= self.record_video, args = (name, framerate, stop_recording))
            # print("Here")
            recorder_thread.start()
            
        self.robot.move(push_motion, push_constraint)
        time.sleep(1)
        self.robot.move(return_motion)
        
        if record:
            recording = False
            recorder_thread.join()

    def save_frame(self, annotation):
        frame = self.cam.get_raw_image()
        big_frame = cv.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
        path = ["frames/img-", str(int(time.time() * 1000)), annotation, ".jpg"]
        #path = ["frames/img-", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), annotation, ".jpg"]
        cv.imwrite("".join(path), frame)

    def show_frame(self):
        frame = self.cam.get_raw_image()
        plt.imshow(frame)
        plt.show()

    def record_frame_series(self, series_name, framerate, stop):
        print("There")
        T = 1/framerate
        while not stop():
            print("recording")
            self.save_frame(series_name)
            time.sleep(T)

    def record_video(self, name, framerate = 0, stop):
        frames = []
        start = time.time()
        while not stop():
            ret, frame = self.cv_cam.read()
            if ret:
                frames.append(cv.resize(frame, (320,240)))
        duration = time.time() - start()
        framerate = len(frames) / duration

        path =  ["videos/vid-", name, "_", datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"), ".mov"]
        # path =  ["videos/vid-", name, "_", str(int(time.time() * 1000)), ".mov"]
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        out = cv.VideoWriter("".join(path) , fourcc, framerate, (320,240), isColor = True)
        for frame in frames:
            out.write(frame)


if __name__ == '__main__':
    test_object_pos = {
        "hardness1": np.array([0,0]),
        "hardness2": np.array([1,0]),
        "hardness3": np.array([2,0]),
        "hardness4": np.array([0,1]),
        "hardness5": np.array([0,0])
    }

    force_thresholds =  [1,3,5,7]
    num_press = 2
    trial_time = "surface"

    
    exp = GridExplorer()
    for key in test_object_pos:
        for _ in range(num_press):
            for force in force_thresholds:
                exp.goto(test_object_pos[key])
                exp.push(force, record = True, name = "".join([key, "_", force, "N"]))
    exp.home()
#
