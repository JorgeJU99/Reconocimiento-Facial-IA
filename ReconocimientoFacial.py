import cv2
import os

# Cambia a la ruta donde hayas almacenado Data
dataPath = 'D:/Proyectos/RECONOCIMIENTO FACIAL/data/'
imagePaths = os.listdir(dataPath)
print('imagePaths=', imagePaths)

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
#face_recognizer.read('D:/Proyectos/RECONOCIMIENTO FACIAL/modelos/modeloEigenFace.xml')
#face_recognizer.read('D:/Proyectos/RECONOCIMIENTO FACIAL/modelos/modeloFisherFace.xml')
face_recognizer.read('D:/Proyectos/RECONOCIMIENTO FACIAL/modelos/modeloLBPHFace.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('D:/Proyectos/RECONOCIMIENTO FACIAL/videos/Dayan.mp4')

if not cap.isOpened():
    raise Exception("Could not open video device")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

faceClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    frame = cv2.normalize(
        frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )

    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y-5),
                    1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        #modeloEigenFace 5700 
        # 500
        # LBPHFace
        if result[1] < 70:
            cv2.putText(frame, '{}'.format(
                imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y-20), 2,
                        0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Reconocimiento facial', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
