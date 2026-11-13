# pylint:disable=no-member

import cv2 as cv

# Read in an image
img = cv.imread("resources/photos/park.jpg")


# Display the image
cv.imshow("Park", img)


# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)


# Wait indefinitely until any key is pressed
cv.waitKey(0)
cv.destroyAllWindows()
