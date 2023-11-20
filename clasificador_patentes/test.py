
import cv2

imagen_path = 'Patentes/Patentes/img01.png'
image = cv2.imread(imagen_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.blur(gray,(3,3))
canny = cv2.Canny(gray_blur,200,200)
canny = cv2.dilate(canny,None,iterations=3)
canny_150 = cv2.Canny(gray_blur,100,200)
canny_150 = cv2.dilate(canny_150,None,iterations=3)
cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    area = cv2.contourArea(c)

    x, y, w, h = cv2.boundingRect(c)
    epsilon = 0.09 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if area > 2000 and area < 8000:
        print('area=', area)
        aspect_ratio = float(w) / h
        if aspect_ratio > 2.4:
            placa = gray[y:y + h, x:x + w]
            cv2.imshow('PLACA', placa)
            cv2.moveWindow('PLACA', 780, 10)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow('Image', image)
cv2.moveWindow('Image', 45, 10)

#cv2.imshow('img oroginal', image)
#cv2.imshow('img gray', gray)
#cv2.imshow('img blur', gray_blur)
#cv2.imshow('img canny', canny)
#cv2.imshow('img canny 150', canny_150)
cv2.waitKey(0)
cv2.destroyAllWindows()

