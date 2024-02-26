#! /usr/bin/env python

import rospy

from math import *


import sys
sys.path.insert(0, '/home/ozkan/Desktop/catkin_ws/src/scripts')

from tools.scan_displayer import ScanDisplayer


if __name__ == '__main__':
    
    scan_displayer = ScanDisplayer()
    try:
        scan_displayer.run_scanner()


    except rospy.ROSInterruptException:
        scan_displayer.end_scan()

        pass
