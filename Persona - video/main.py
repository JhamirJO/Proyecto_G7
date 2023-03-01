import cv2 
import time
import argparse
import os

# construir el analizador de argumentos y analizar los argumentos de la línea de comando
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='pedestrians.avi',
                    help='Ruta del video de entrada ')
parser.add_argument('-o', '--output', required=True, help='Ruta del video de salida')
parser.add_argument('-s', '--speed', default='yes', choices=['fast', 'slow'],
                    help='Elegir la velocidad del video')
args = vars(parser.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error mientras se abrio el video. Intente de nuevo...')

# obtener el ancho y la altura del cuadro
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# mantener un tamaño de cuadro mínimo para predicciones precisas
if frame_width < 400: # if image width < 400
        frame_width = 400
        ratio = frame_width / float(frame_width) # encuentra la relacion de ancho a alto
        frame_height = int(frame_width * ratio)
# definir el códec y crear el objeto VideoWriter
out = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))
frame_count = 0
total_fps = 0

# leer hasta el final del video
while (cap.isOpened()):
    # captura cada cuadro del video
    ret, frame = cap.read()
    if ret == True:
        start_time = time.time()

        frame = cv2.resize(frame, (frame_width, frame_height))
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if args['speed'] == 'fast':
            rects, weights = hog.detectMultiScale(img_gray, padding=(4, 4), scale=1.02)
        elif args['speed'] == 'slow':
            rects, weights = hog.detectMultiScale(img_gray, winStride=(4, 4), padding=(4, 4), scale=1.02)
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.13:
                continue
            elif weights[i] < 0.3 and weights[i] > 0.13:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if weights[i] < 0.7 and weights[i] > 0.3:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 122, 255), 2)
            if weights[i] > 0.7:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Confianza alta', (10, 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Confianza media', (10, 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (50, 122, 255), 2)
        cv2.putText(frame, 'Confianza baja', (10, 55), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        frame_write_name = args['input'].split('/')[-1].split('.')[0]
        cv2.imwrite(f"../outputs/frames/{args['speed']}_{frame_write_name}_{frame_count}.jpg", frame)

        # Medir el tiempo transcurrido para las detecciones
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # print(f"{fps:.3f} FPS")
        # agregar al total de FPS
        total_fps += fps
        # añadir al número total de fotogramas
        frame_count += 1
        cv2.imshow("Preview", frame)
        out.write(frame)
        # presiona `q` para salir
        wait_time = max(1, int(fps / 4))
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break


avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}" )
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()


#python main.py --input video.avi --output video_fast.avi --speed fast
#python main.py --input video.avi --output video_slow.avi --speed slow

#python main.py --input video2.mp4 --output video2_fast.mp4 --speed fast
#python main.py --input video2.mp4 --output video2_slow.mp4 --speed slow