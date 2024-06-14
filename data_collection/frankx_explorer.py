from frankx import (Gripper, Robot, LinearMotion, Affine, LinearRelativeMotion, Kinematics, NullSpaceHandling,
                    JointMotion, MotionData, Reaction, Measure, StopMotion)
import numpy as np
import time
import os
from gelsight import gsdevice
import cv2 as cv
from datetime import datetime
from matplotlib import pyplot as plt
from threading import Thread

import uuid
import json


class GridExplorer():
    def __init__(self):
        self.robot = Robot("172.168.0.2")
        self.robot.set_default_behavior()  
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(0.05)

        #self.cam = gsdevice.Camera(gsdevice.Finger.MINI, 0)
        #self.cam.connect()
        self.cv_cam = cv.VideoCapture(0)
        #self.res = (0, 0, 320, 240)
        #print('image size = ', self.cam.get_raw_image().shape)

        #self.neutral_joints = [0.6118366520437153, 0.15716609762216865, -0.25319002091814896, -2.394232290291543, 0.029041619664173065, 2.5748198744888313, 1.1364431301769282]    # Calibration 25.11.2022
        #self.neutral_joints = [0.3707151330437521, 0.16326812298140542, -0.028099087464621014, -2.3989490835251646, 0.00702887462245093, 2.5924458934995864, 1.1128487507518794]    # Calibration 28.11.2022
        #self.neutral_joints = [0.37210778693148966, 0.28854662857557595, -0.02800002464495207, -2.39302845971585, 0.01651057972216312, 2.7108360546139134, 1.106010951997505]
        #self.neutral_joints = [0.3742012436992107, 0.3965798386015373, -0.027704081975832653, -2.373704329876077, 0.028091501702229765, 2.7978853032960225, 1.097400190761943]
        
        #high position for new test objects
        self.neutral_joints = [0.37087141247348876, 0.1417811051004933, -0.027766077848984373, -2.3990414049764155, 0.0059387914338375125, 2.5672801672813654, 1.1137218490452334]
        self.homing_motion = JointMotion(self.neutral_joints)

        self.cruise_speed = 0.1

        self.grid_size_x = 4
        self.grid_size_y = 5 

        self.cell_size = 0.09       

        self.pos = np.array([0,0])
        self.home()
        

    def home(self):
        self.robot.set_dynamic_rel(self.cruise_speed)
        self.robot.move(self.homing_motion)
        self.pos = np.array([0,0])

    def goto(self, pos_):
        if not (pos_[0] < self.grid_size_x and pos_[1] < self.grid_size_y):
            return
        if np.linalg.norm(pos_ - self.pos) < 0.01:
            return
        rel = pos_ - self.pos
        motion = LinearRelativeMotion(Affine(rel[0]*self.cell_size, -rel[1]*self.cell_size, 0.0))
        self.robot.set_dynamic_rel(self.cruise_speed)
        self.robot.move(motion)
   

        self.pos = pos_

    def push(self, max_force, vel = 0.05, noise = False, object_name = "default", record = False): # force should be positive even though negative z direction
        self.robot.set_dynamic_rel(vel)
        max_force = np.abs(max_force) # just to be safe
      
        
        return_q = self.robot.get_state().q
        return_motion =  JointMotion(return_q)
        after_push_record_motion =  LinearRelativeMotion(Affine(0.0, 0.0, 0.01)) #
        
        push_constraint = MotionData().with_reaction(Reaction(Measure.ForceZ < -max_force, StopMotion()))
        after_push_constaint = MotionData().with_reaction(Reaction(Measure.ForceZ > 5000, StopMotion()))
        push_motion = LinearRelativeMotion(Affine(0.0, 0.0, -0.12))

        print("bla")

        if record:
            recording = True
            stop_recording = lambda: not recording
            recorder_thread = Thread(target= self.record, args = (object_name, force, vel, stop_recording))
            recorder_thread.start()

        if noise:
            x_off = -0.02 + np.random.random() * 0.04
            y_off = -0.02 + np.random.random() * 0.04
            noise_movement = LinearRelativeMotion(Affine(x_off, y_off, 0.00))
            self.robot.move(noise_movement)
            
        self.robot.move(push_motion, push_constraint)
        self.robot.move(after_push_record_motion,after_push_constaint)
        self.robot.move(return_motion)
        
        if record:
            recording = False
            recorder_thread.join()


    def record(self, object_name, force, vel, stop):
        if os.path.exists("tmp/numerical.txt"):
            os.remove("tmp/numerical.txt")

        id = uuid.uuid4()

        frames = []
        start = time.time()

        while not stop():
            ret, frame = self.cv_cam.read()
            if ret:
                timestamp = int(time.time() * 1000)
                img = cv.resize(frame, (320,240))
                frames.append( (timestamp, img))

        duration = time.time() - start
        actual_framerate = len(frames) / duration


        #video
        path =  ["/home/ip-hard/Projects/ip-hardness/data_collection/data/videos/", str(id), ".mov"]
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        out = cv.VideoWriter("".join(path) , fourcc, actual_framerate, (320,240), isColor = True)
        for f in frames:
            out.write(f[1])

        #meta
        meta = {
            "id": str(id),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object": object_name,
            "max_force": force,
            "max_vel": vel
        }
        #meta
        with open("data/meta/"+ str(id) + ".json", "w") as f:
            json.dump(meta, f)


        #frames
        time_array = np.zeros(len(frames))
        frame_matrix = np.zeros((len(frames), 240, 320, 3))

        for i,(t, f) in enumerate(frames):
            time_array[i] = t
            frame_matrix[i,:,:,:] = f

        with open("data/frames/" + str(id) + "_times.npy", "wb") as f:
            np.save(f, time_array)

        with open("data/frames/" + str(id) + "_frames.npy", "wb") as f:
            np.save(f, frame_matrix)



        numerical =  np.loadtxt("../tmp/numerical.txt")
        #forces
        with open("data/forces/" + str(id) + ".npy", 'wb') as f:
            np.save(f, numerical[:, [0,1]])
        #angles

        with open("data/angles/" + str(id) + ".npy", 'wb') as f:
            np.save(f, numerical[:, [0,2,3,4,5,6,7,8]])

        """
        poses = np.apply_along_axis(
            lambda q: Kinematics.forward(q),
            1,
            numerical[:, [2,3,4,5,6,7,8]]
        )
        with open("data/poses/" + str(id) + ".npy", "wb") as f:
            np.save(f, poses)
        
        
        
        """


        


if __name__ == '__main__':
    test_object_pos = {
        "lego": np.array([0,0]),
        "foam": np.array([1,0]),
        #"hardness3": np.array([2,0]),
        "banana": np.array([0,1]),
        "ball": np.array([1,1])
        
    }

    min_vel = 0.02
    max_vel = 0.05

    min_force = 1
    max_force = 5

    collection_epochs = 10
    
    exp = GridExplorer()

    # stay closer to the workspaces to make pushes faster
    #exp.robot.move(LinearRelativeMotion(Affine(0.0, 0.0, -0.04)))

    # exp.push(3, record = True, name = "".join(["3N_test"]))

    for _ in range(collection_epochs):
        for key in test_object_pos:
            force = min_force + np.random.random() * (max_force-min_force) 
            vel = min_vel + np.random.random() * (max_vel - min_vel)
            exp.goto(test_object_pos[key])
            print(_, key, force, vel)
            exp.push(force, vel = vel, noise = False, record = True, object_name= key)
        exp.home()
