import cv2

working_cameras = []
for i in range(0, 3):
    test_feed = cv2.VideoCapture(i)
    success, frame = test_feed.read()

    if success:
        print(f"Camera index{i} is functional.")
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

    for i in range(0, no_of_cameras):
        flag, frame = camera_feeds[i].read()
        if flag:
            cv2.imshow('Feed ' + str(i+1), frame)

    if cv2.waitKey(20) == ord('q'):
        for i in range(0, no_of_cameras):
            camera_feeds[i].release()
        cv2.destroyAllWindows()
        break