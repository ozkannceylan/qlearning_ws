
import rospy
from scripts.old.mp2 import MotionPlanner    



if __name__ == '__main__':
    control=MotionPlanner()
    try:

        control.run()
    except rospy.ROSInterruptException:
        control.end_sim()
        print("Simulation is terminated.")