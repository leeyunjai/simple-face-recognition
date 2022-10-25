import cv2,dlib
import numpy as np

if __name__ == "__main__":
  detector = cv2.CascadeClassifier("model/lbpcascade_frontalface_improved.xml")
  predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
  #predictor = dlib.shape_predictor("model/shape_predictor_5_face_landmarks.dat")

  cap = cv2.VideoCapture(0)
  while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rects:
      rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
      shape = predictor(gray, rect)

      coords = np.zeros((shape.num_parts, 2), dtype="int")
      for i in range(0, shape.num_parts):
          coords[i] = (shape.part(i).x, shape.part(i).y)

      for (i, (x, y)) in enumerate(coords):
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(img, str(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow('face_landmark', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cv2.destroyAllWindows()
