import pickle
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--record', help='trained file', default='pretrained/record')
  args = parser.parse_args()

  with open(args.record, "rb") as f:
    known_faces = pickle.load(f)

    for _name, _value in zip(known_faces[0], known_faces[1]):
      print(_name, _value)

