import cv2,dlib,argparse
import numpy as np
import pickle

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--record', help='record', default='pretrained/record')
  args = parser.parse_args()

  with open(args.record, "rb") as f :
    known_faces = pickle.load(f)

  detector = cv2.CascadeClassifier("model/lbpcascade_frontalface_improved.xml")
  face_encoder = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")
  predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
  #predictor = dlib.shape_predictor("model/shape_predictor_5_face_landmarks.dat")

  face_names = []
  face_records = []
  threshold = 0.5

  cap = cv2.VideoCapture(0)
  while True:
    img = cap.read()[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    rects = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rects:
      rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
      shape = predictor(gray, rect)

      face_encoding = np.array(face_encoder.compute_face_descriptor(img, shape, 1))
      name = "Guest"
      matches = []
      matches = list(np.linalg.norm(known_faces[1] - face_encoding, axis=1))
      if min(matches) < threshold:
        name = known_faces[0][matches.index(min(matches))]

      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
      cv2.putText(img, str(name), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.imshow('face_recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()
