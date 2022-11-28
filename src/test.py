from frankx import (Gripper, Robot, LinearMotion, Affine, LinearRelativeMotion, Kinematics, NullSpaceHandling,
                    JointMotion, MotionData, Reaction, Measure, StopMotion)
import time
 
robot = Robot("172.168.0.2")
gripper= Gripper("172.168.0.2")
robot.set_default_behavior()
robot.recover_from_errors()
robot.set_dynamic_rel(0.1)






neutral_joint_space = [0.6118366520437153, 0.15716609762216865, -0.25319002091814896, -2.394232290291543, 0.029041619664173065, 2.5748198744888313, 1.1364431301769282]
homing_motion = JointMotion(neutral_joint_space)

#x forward
#y right
#z downward



reaction_motion = LinearRelativeMotion(Affine(0.0, 0.0, 0.0)) 
push_constraint = MotionData().with_reaction(Reaction(Measure.ForceZ < -5, StopMotion()))

motion = LinearRelativeMotion(Affine(0.0, 0.0, 0.15))
robot.move(motion, push_constraint)



