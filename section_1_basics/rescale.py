# pylint:disable=no-member

import cv2 as cv


def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def changeRes(width, height):
    # Live video
    capture.set(3, width)
    capture.set(4, height)


# Reading Videos
capture = cv.VideoCapture("resources/videos/dog.mp4")

while True:
    isTrue, frame = capture.read()

    if isTrue:
        cv.imshow("Video", frame)
        frame_resized = rescaleFrame(frame, scale=0.2)
        cv.imshow("Video Resized", frame_resized)
        if cv.waitKey(20) & 0xFF == ord("d"):
            # press d to quit video
            break
    else:
        break

capture.release()
cv.destroyAllWindows()
