import cv2
import dlib
import numpy as np
import sys
import pickle

known_face_encodings = []
known_face_names = []
known_faces = [known_face_names, known_face_encodings]

def show_usage():
    print('[!] Usage: python train.py')
    print('\t train.py [Face_File] [Face Name] [Trained File] [init/add]')

if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        show_usage()
        exit() 

    detector = cv2.CascadeClassifier("model/lbpcascade_frontalface_improved.xml")
    predictor = dlib.shape_predictor("model/shape_predictor_5_face_landmarks.dat")
    face_encoder = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")

    if sys.argv[4] == "add":
        with open(sys.argv[3], "rb") as f :
            known_faces = pickle.load(f)

    img = cv2.imread(sys.argv[1])    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    rects = detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)

        face_encoding = np.array(face_encoder.compute_face_descriptor(img, shape, 1))
        
        known_faces[0].append(sys.argv[2])
        known_faces[1].append(face_encoding)

    with open(sys.argv[3], "w+b") as f:
        pickle.dump(known_faces, f)
