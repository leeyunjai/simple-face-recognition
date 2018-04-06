import pickle
import sys
def show_usage():
    print('[!] Usage: python dbtest.py')
    print('\t dbtest.py [Trained File]')

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        show_usage()
        exit() 

    with open(sys.argv[1], "rb") as f :
        known_faces = pickle.load(f)
        
    for known_face_name in known_faces[0]:
        print(known_face_name)
