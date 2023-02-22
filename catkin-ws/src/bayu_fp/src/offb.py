#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

import numpy as np
import cv2
import cv2.aruco as aruco

current_state = State()

def state_cb(msg):
    global current_state
    current_state = msg

def calibrate():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((10*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    image = cv2.imread('./checkerboard.jpg')
    cap = cv2.VideoCapture(0)
    found = 0

    while found < 10:
        ret, img = cap.read()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (10, 7), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(image, (10, 7), corners2, ret)
            found += 1
        
        cv2.imshow('img', image)

    cap.release()
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

if __name__ == "__main__":
    rospy.init_node("offb_node_py")

    state_sub = rospy.Subscriber("mavros/state", State, callback = state_cb)

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    local_vel_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)


    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)    

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while(not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()

    pose = PoseStamped()

    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 2

    vel = Twist()

    vel.linear.x = 0
    vel.linear.y = 0
    vel.linear.z = 0

    # Send a few setpoints before starting
    for i in range(100):   
        if(rospy.is_shutdown()):
            break

        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    '''           '''
    ''' DETECTION '''
    '''           '''
    # initialize video stream
    cap = cv2.VideoCapture(0)

    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    arucoParams = aruco.DetectorParameters()

    # load camera calibration
    # get camera matrix and distortion
    ret, mtx, dist, rvecs, tvecs = calibrate()

    # connection state
    connected = False
    # connectio progress
    progress = 0

    # array of proximity
    tprox = []
    wprox = []
    hprox = []

    # controller center
    tcen = []

    # controller threshold
    width = 0
    height = 0

    while(not rospy.is_shutdown()):
        if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if(set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")
            
            last_req = rospy.Time.now()
        else:
            if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if(arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")
                    local_pos_pub.publish(pose)
            
                last_req = rospy.Time.now()
        
            elif(current_state.armed):
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                corners, ids, rejected = aruco.detectMarkers(gray, arucoDict,
                    parameters=arucoParams)

                # verify *at least* one ArUco marker was detected
                if len(corners) > 0:

                    # flatten the ArUco IDs list
                    ids = ids.flatten()
                    

                    ''' POSE ESTIMATION '''
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
                    (rvec-tvec).any()

                    for i in range(rvec.shape[0]):
                        # draw local axes on each ArUco
                        cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                        # draw bounding box of each ArUco
                        aruco.drawDetectedMarkers(frame, corners)

                    # loop over the detected ArUCo corners
                    for (markerCorner, markerID) in zip(corners, ids):
                        # extract the marker corners (which are always returned in
                        # top-left, top-right, bottom-right, and bottom-left order)
                        corner = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corner
                        # convert each of the (x, y)-coordinate pairs to integers
                        topRight = (int(topRight[0]), int(topRight[1]))
                        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                        topLeft = (int(topLeft[0]), int(topLeft[1]))


                        # compute and draw the center (x, y)-coordinates of the ArUco marker
                        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                        w = bottomRight[0] - topLeft[0]
                        h = bottomRight[1] - topLeft[1]

                        cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
                        # draw the ArUco marker ID on the image
                        cv2.putText(frame, str(markerID),
                            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

                    if not connected:
                        # increment progress
                        progress += 1
                        # if progress reach 100, then connect
                        if progress >= 100:
                            connected = True

                            ''' Translation '''
                            sum = (0, 0)
                            sum2 = 0

                            for t in tprox:
                                sum = (sum[0] + t[0], sum[1] + t[1])
                            
                            tcen = (sum[0]/len(tprox), sum[1]/len(tprox))
                            
                            for w in wprox:
                                sum2 += w
                            
                            width = sum2/len(wprox)
                            sum2 = 0

                            for h in hprox:
                                sum2 += h
                            
                            height = sum2/len(hprox)

                            # clear proximation
                            tprox.clear()
                            wprox.clear()
                            hprox.clear()

                        tprox.append((cX, cY))
                        wprox.append(w)
                        hprox.append(h)

                    # if disturbed while trying to disconnect,
                    # then reset
                    else:
                        progress = 100
                        
                        # Translation velocity x
                        if(cX < tcen[0] - 50):
                            vel.linear.x = -1
                        elif(cX > tcen[0] + 50):
                            vel.linear.x = 1
                        else:
                            vel.linear.x = 0

                        # Translation velocity y
                        if(cY < tcen[1] - 50):
                            vel.linear.z = 1
                        elif(cY > tcen[1] + 50):
                            vel.linear.z = -1
                        else:
                            vel.linear.z = 0

                        # Translation velocity z
                        if(w < width - 10):
                            vel.linear.y = -1
                        elif(w > width + 10):
                            vel.linear.y = 1
                        else:
                            vel.linear.y = 0

                # if there is no ArUco detected
                else:
                    # if no ArUco detected when connected
                    if connected:
                        # decrement progess
                        progress -= 1
                        # id progress reach 0, then disconnect
                        if progress <= 0:
                            connected = False

                            # reset velocity
                            vel.linear.x = 0
                            vel.linear.y = 0
                            vel.linear.z = 0
                    
                    # if disturbed while connecting,
                    # then reset progress
                    else:
                        progress = 0

                        # reset proximation
                        tprox.clear()
                        wprox.clear()
                        hprox.clear()

                # clamp value to limit progress value between 0 to 100
                progress = max(min(progress, 100), 0)

                cv2.putText(frame, 'Connection = %i' % (progress), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0, 0), 2)
                cv2.putText(frame, 'Connected = ' + str(connected), (0, 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                cv2.putText(frame, 'Vel = (%i, %i, %i)' % (vel.linear.x, vel.linear.y, vel.linear.z), (0, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2)

                # display video stream
                cv2.imshow('frame', frame)

                # press q to close program
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        local_vel_pub.publish(vel)

        rate.sleep()