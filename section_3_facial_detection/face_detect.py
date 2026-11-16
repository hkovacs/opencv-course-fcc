# pylint:disable=no-member

import cv2 as cv

haar_cascade = cv.CascadeClassifier("section_3_facial_detection/haar_face.xml")

img_lady = cv.imread("resources/photos/lady.jpg")
faces_rect_lady = haar_cascade.detectMultiScale(img_lady, scaleFactor=1.1, minNeighbors=3)
for x, y, w, h in faces_rect_lady:
    cv.rectangle(img_lady, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
cv.imshow("Lady", img_lady)

gray_lady = cv.cvtColor(img_lady, cv.COLOR_BGR2GRAY)
faces_rect_gray_lady = haar_cascade.detectMultiScale(gray_lady, scaleFactor=1.1, minNeighbors=3)
for x, y, w, h in faces_rect_gray_lady:
    cv.rectangle(gray_lady, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
cv.imshow("Gray Lady", gray_lady)

img = cv.imread("resources/photos/group-1.jpg")
faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1)
for x, y, w, h in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
cv.imshow("Group of people", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)
for x, y, w, h in faces_rect:
    cv.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
cv.imshow("Gray People", gray)

print(f"Number of faces found = {len(faces_rect)}")

for x, y, w, h in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
cv.imshow("Detected Faces", img)


cv.waitKey(0)
