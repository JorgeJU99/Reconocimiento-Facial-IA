import cv2
import os
import numpy as np

dataPath = 'D:/Proyectos/RECONOCIMIENTO FACIAL/data/' #Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
	label = label + 1


# Métodos para entrenar el reconocedor
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.save('D:/Proyectos/RECONOCIMIENTO FACIAL/modelos/modeloEigenFace.xml')
#face_recognizer.save('D:/Proyectos/RECONOCIMIENTO FACIAL/modelos/modeloFisherFace.xml')
face_recognizer.save('D:/Proyectos/RECONOCIMIENTO FACIAL/modelos/modeloLBPHFace.xml')
print("Modelo almacenado...")