# pylint:disable=no-member

import cv2 as cv

# Read in an image
img = cv.imread("resources/photos/park.jpg")


# Display the image
cv.imshow("Park", img)


# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)


# Blur the image
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
cv.imshow("Blur", blur)


# Edge Cascade
canny = cv.Canny(blur, 125, 175)
cv.imshow("Canny Edges", canny)


# Wait indefinitely until any key is pressed
cv.waitKey(0)
cv.destroyAllWindows()
