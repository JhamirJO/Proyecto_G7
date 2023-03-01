import cv2
import glob as glob
import os

#inicializamos el HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image_paths = glob.glob('test2.jpg')

for image_path in image_paths:
    image_name = image_path.split(os.path.sep)[-1]
    image = cv2.imread(image_path)

    # Mantenemos un tamaño de imagen mínimo para predicciones precisas
    if image.shape[1] < 400:  # si el ancho de la imagen < 400
        (height, width) = image.shape[:2]
        ratio = width / float(width)  # encontrar la relacion de del alto y ancho
        image = cv2.resize(image, (400, width * ratio))  # cambiamos el tamaño de la imagen de acuerdo con la relacion ancho-algo
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects, weights = hog.detectMultiScale(img_gray, winStride=(2, 2), padding=(20, 20), scale=1.04)

    for i, (x, y, w, h) in enumerate(rects):
        if weights[i] < 0.13:
            continue
        elif weights[i] < 0.3 and weights[i] > 0.13:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if weights[i] < 0.7 and weights[i] > 0.3:
            cv2.rectangle(image, (x, y), (x + w, y + h), (50, 122, 255), 2)
        if weights[i] > 0.7:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(image, 'Confianza alta', (10, 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, 'Confianza moderada', (10, 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 122, 255), 2)
    cv2.putText(image, 'Confianza baja', (10, 75), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('HOG detection', image)
    cv2.imwrite(f"../outputs/{image_name}", image)
    cv2.waitKey(0)
