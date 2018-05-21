# Test Environment
- Raspberrypi 3B (Jessie) or Ubuntu 16.04/17.10
- python 3.4~ / dlib 19.10 / opencv 3.3~ (I think the slightly lower version will not matter.)
- I got test images from google free images.
- ageitgey/face_recognition and pyimagesearch.com (Reference)

# File list
- python3 dbtest.py [Trained File]  # Check Pretrained DB
- python3 train.py [Face_File] [Face Name] [Trained File] [init/add] # Train face - name (init - reset / add - append)
- python3 landmark.py # Demo face landmark detection
- python3 demo.py [Trained File] # Demo face recognition
