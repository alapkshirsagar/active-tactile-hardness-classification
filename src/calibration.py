from frankx import (Gripper, Robot, LinearMotion, Affine, LinearRelativeMotion, Kinematics, NullSpaceHandling,
                    JointMotion, MotionData, Reaction, Measure, StopMotion)
import time

robot = Robot("172.168.0.2")
robot.set_default_behavior()
robot.recover_from_errors()
robot.set_dynamic_rel(0.1)

reaction_motion = LinearRelativeMotion(Affine(0.0, 0.0, 0.0)) 
push_constraint = MotionData().with_reaction(Reaction(Measure.ForceZ < -5, StopMotion()))

motion = LinearRelativeMotion(Affine(0.0, 0.0, 0.15))
#robot.move(motion, push_constraint)

print(robot.get_state().q)
