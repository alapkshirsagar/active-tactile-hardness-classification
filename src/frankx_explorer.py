from frankx import (Gripper, Robot, LinearMotion, Affine, LinearRelativeMotion, Kinematics, NullSpaceHandling,
                    JointMotion, MotionData, Reaction, Measure, StopMotion)

import numpy as np
import time




class GridExplorer():
    def __init__(self):
        self.robot = Robot("172.168.0.2")
        self.robot.set_default_behavior()  
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(0.05)

        #self.neutral_joints = [0.6118366520437153, 0.15716609762216865, -0.25319002091814896, -2.394232290291543, 0.029041619664173065, 2.5748198744888313, 1.1364431301769282]    # Calibration 25.11.2022
        self.neutral_joints = [0.3707151330437521, 0.16326812298140542, -0.028099087464621014, -2.3989490835251646, 0.00702887462245093, 2.5924458934995864, 1.1128487507518794]    # Calibration 28.11.2022
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

    def push(self, max_force): # force should be positive even though negative z direction
        max_force = np.abs(max_force) # just to be safe
      
        
        return_q = self.robot.get_state().q
        return_motion = JointMotion(return_q)
        
        push_constraint = MotionData().with_reaction(Reaction(Measure.ForceZ < -max_force, StopMotion()))
        push_motion = LinearRelativeMotion(Affine(0.0, 0.0, -0.15))
        self.robot.move(push_motion, push_constraint)

        self.robot.move(return_motion)

        #TODO: implement force recording, see
        """
        https://git.ias.informatik.tu-darmstadt.de/tactile-sensing/interpretable-representations/-/blob/master/src/visuotactile_sensor/src/robot_control/robot_control.cpp#L294
        """

        



if __name__ == '__main__':
    #time.sleep(10)
    exp = GridExplorer()
    exp.goto(np.array([0,0]))
    exp.push(1)
    exp.goto(np.array([1,0]))
    exp.push(1)
    exp.goto(np.array([2,0]))
    exp.push(1)
    exp.goto(np.array([0,1]))
    exp.push(1)
    exp.goto(np.array([1,1]))
    exp.push(1)
    exp.home()
