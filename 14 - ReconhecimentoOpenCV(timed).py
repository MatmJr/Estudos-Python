import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np
import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def contador(facesDetectadas):
    for face in facesDetectadas:
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorFacial = [fd for fd in descritorFacial]
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        print("Distâncias: {}".format(distancias))
        minimo = np.argmin(distancias)
        print(minimo)
        distanciaMinima = distancias[minimo]
        print(distanciaMinima)

        if distanciaMinima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
        else:
            nome = ' '

        #cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
        #texto = "{} {:.4f}".format(nome, distanciaMinima)
        #cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))

    #cv2.imshow("Detector hog", imagem)
    #cv2.waitKey(0)

#cv2.destroyAllWindows()




file = open("database.txt", "w")


detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("recursos/indices.pickle", allow_pickle=True)
descritoresFaciais = np.load("recursos/descritores.npy")
limiar = 0.58

for arquivo in glob.glob(os.path.join("Reconhecimento", "*.jpg")):
    imagem = cv2.imread(arquivo)
    facesDetectadas_aux = detectorFace(imagem, 3)

    wrapped = wrapper(contador, facesDetectadas_aux)
    file.write(str(timeit.timeit(wrapped, number=1)) + "\n")

file.close()




