import cv2,dlib,argparse
import numpy as np
import pickle

known_faces = [[],[]]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--image', help='image file', required=True)
  parser.add_argument('--name', help='name', required=True)
  parser.add_argument('--record', help='trained file', required=True)
  parser.add_argument('--mode', help='init | add', choices=['init', 'add'], required=True)
  args = parser.parse_args()

  detector = cv2.CascadeClassifier("model/lbpcascade_frontalface_improved.xml")
  face_encoder = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")
  predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
  #predictor = dlib.shape_predictor("model/shape_predictor_5_face_landmarks.dat")

  if args.mode == "add":
    with open(args.record, "rb") as f :
      known_faces = pickle.load(f)

  img = cv2.imread(args.image)  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  rects = detector.detectMultiScale(gray, 1.3, 5)

  for (x, y, w, h) in rects:
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    shape = predictor(gray, rect)

    face_encoding = np.array(face_encoder.compute_face_descriptor(img, shape, 1))

    known_faces[0].append(args.name)
    known_faces[1].append(face_encoding)

  with open(args.record, "w+b") as f:
    pickle.dump(known_faces, f)
