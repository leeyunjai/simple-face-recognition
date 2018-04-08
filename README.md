# Test Environment
- Raspberrypi 3B (Jessie) or Ubuntu 17.10
- python3
- dlib 19.10 (I think the slightly lower  version will not matter.)
- opencv 3.4 (I think the slightly lower version will not matter.)
- Test image got from google free images.
- ageitgey/face_recognition and pyimagesearch.com (Reference)

# File list
- python3 dbtest.py [Trained File]  # Check Pretrained DB
- python3 train.py [Face_File] [Face Name] [Trained File] [init/add] # Train face - name
- python3 landmark.py # Demo face landmark detection
- python3 demo.py [Trained File] # Demo face recognition
