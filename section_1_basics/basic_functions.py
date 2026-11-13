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


# Dilate the image
dilated = cv.dilate(canny, (7, 7), iterations=3)
cv.imshow("Dilated", dilated)


# Erode the image
eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow("Eroded", eroded)


# Resize the image
resized = cv.resize(img, (770, 770), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized", resized)


# Crop the image
cropped = img[50:200, 200:400]
cv.imshow("Cropped", cropped)


# Wait indefinitely until any key is pressed
cv.waitKey(0)
cv.destroyAllWindows()
