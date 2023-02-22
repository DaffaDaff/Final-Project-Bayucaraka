import numpy as np
import cv2
import cv2.aruco as aruco
import yaml
import calibration

def run():
    # initialize video stream
    cap = cv2.VideoCapture(0)

    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    arucoParams = aruco.DetectorParameters()

    # load camera calibration
    # get camera matrix and distortion
    ret, mtx, dist, rvecs, tvecs = calibration.calibrate()

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

    # velocity
    tvel = [0, 0, 0]

    while(True):
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
                    tvel[0] = -1
                elif(cX > tcen[0] + 50):
                    tvel[0] = 1
                else:
                    tvel[0] = 0

                # Translation velocity y
                if(cY < tcen[1] - 50):
                    tvel[1] = -1
                elif(cY > tcen[1] + 50):
                    tvel[1] = 1
                else:
                    tvel[1] = 0

                # Translation velocity z
                if(w < width - 10):
                    tvel[2] = -1
                elif(w > width + 10):
                    tvel[2] = 1
                else:
                    tvel[2] = 0

        # if there is no ArUco detected
        else:
            # if no ArUco detected when connected
            if connected:
                # decrement progess
                progress -= 1
                # id progress reach 0, then disconnect
                if progress <= 0:
                    connected = False
            
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

        cv2.putText(frame, str(progress), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2)
        cv2.putText(frame, str(connected), (50, 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2)

        # display video stream
        cv2.imshow('frame', frame)

        # press q to close program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()