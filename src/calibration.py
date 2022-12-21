from frankx import (Gripper, Robot, LinearMotion, Affine, LinearRelativeMotion, Kinematics, NullSpaceHandling,
                    JointMotion, MotionData, Reaction, Measure, StopMotion)
import time

robot = Robot("172.168.0.2")
robot.set_default_behavior()
robot.recover_from_errors()
robot.set_dynamic_rel(0.1)

neutral_joints = [0.37210778693148966, 0.28854662857557595, -0.02800002464495207, -2.39302845971585, 0.01651057972216312, 2.7108360546139134, 1.106010951997505]
homing_motion = JointMotion(neutral_joints)
#robot.move(homing_motion)

motion = LinearRelativeMotion(Affine(0.0, 0.0, -0.01))
robot.move(motion)

print(robot.get_state().q)
