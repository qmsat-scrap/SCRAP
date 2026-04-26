import cv2
import time
import math
from datetime import datetime

working_cameras = []
for i in range(0, 3):
    test_feed = cv2.VideoCapture(i)
    success, frame = test_feed.read()

    if success:
        print(f"Camera index {i} is functional.")
        working_cameras.append(i)
    else:
        print(f"ERROR: Camera index {i} is not functional.")
    
    test_feed.release()


if len(working_cameras) == 0:
    raise Exception("No working cameras found")
else:
    print(f"Proceeding with {len(working_cameras)} working cameras...")

camera_feeds = [cv2.VideoCapture(i) for i in working_cameras]
no_of_cameras = len(camera_feeds)
while True:
    start = time.time()

    for i in range(0, no_of_cameras):
        flag, frame = camera_feeds[i].read()
        if flag:
            end = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            filename = f"cam{i}_{timestamp}.jpg" #have separate folder for each cam?

            cv2.imwrite(filename, frame)
            cv2.putText(frame, 
                        "FPS: " + str(math.ceil(1 / (end - start))),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA)    # FPS counter, just for observation         
            cv2.imshow('Feed ' + str(i+1), frame)


    if cv2.waitKey(20) == ord('q'):
        for i in range(0, no_of_cameras):
            camera_feeds[i].release()
        cv2.destroyAllWindows()
        break