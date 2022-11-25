from frankx import (Gripper, Robot, LinearMotion, Affine, LinearRelativeMotion, Kinematics, NullSpaceHandling,
                    JointMotion, MotionData, Reaction, Measure, StopMotion)

import numpy as np
import time


class GridExplorer():
    def __init__(self):
        self.robot = Robot("172.168.0.2")
        self.robot.set_default_behavior()  
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(0.1)

        self.neutral_joints = [0.6118366520437153, 0.15716609762216865, -0.25319002091814896, -2.394232290291543, 0.029041619664173065, 2.5748198744888313, 1.1364431301769282]
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

        print(self.pos, pos_)
    
        if not (pos_[0] < self.grid_size_x and pos_[1] < self.grid_size_y):
            return
        if np.linalg.norm(pos_ - self.pos) < 0.01:
            return

     
        rel = pos_ - self.pos
        motion = LinearRelativeMotion(Affine(rel[0]*self.cell_size, -rel[1]*self.cell_size, 0.0))
        self.robot.move(motion)

        self.pos = pos_

    def push(self,force):
        pass


if __name__ == '__main__':
    exp = GridExplorer()
   
    for y in range(5):
        for x in range(4):
            exp.goto(np.array([x,y]))
            time.sleep(0.5)
       
