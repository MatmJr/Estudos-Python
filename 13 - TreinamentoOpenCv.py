import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritoresFaciais = None

# Importante: percorer todos os arquivos de uma pasta!!! bib: glob serve para percorrer uma pasta atrás de determinado formato
for arquivo in glob.glob(os.path.join("Treinamento","*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas = detectorFace (imagem, 1)
    numeroFacesDetectadas = len(facesDetectadas)

    if numeroFacesDetectadas > 1:
        print("Há mais de uma face na imagem {}".format(arquivo))
        exit(0)
    elif numeroFacesDetectadas < 1:
        print("Nenhuma face encontrada no arquivo {}".format(arquivo))
        exit(0)

    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)

        listaDescritorFacial = [df for df in descritorFacial]
        nparrayDescritorFacial = np.asanyarray(listaDescritorFacial, dtype=np.float64)

        nparrayDescritorFacial = nparrayDescritorFacial[np.newaxis, :]

        if descritoresFaciais is None:
            descritoresFaciais = nparrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, nparrayDescritorFacial), axis = 0)

        indice[idx] = arquivo
        idx += 1

np.save("recursos/descritores.npy", descritoresFaciais)
with open("recursos/indices.pickle", 'wb') as f:
    cPickle.dump(indice, f)

