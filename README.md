# Test Environment
- Raspberrypi 3B, 4B (Jessie+) or Ubuntu
- python 3.3+ / dlib 19.6+ / opencv 3.3+ (I think the slightly lower version will not matter.)
- I got test images from google free images.
- ageitgey/face_recognition and pyimagesearch.com (Reference)

# File list
- python3 dbtest.py [Trained File]  # Check Pretrained DB
- python3 train.py [Face_File] [Face Name] [Trained File] [init/add] # Train face - name (init - reset / add - append)
- python3 landmark.py # Demo face landmark detection
- python3 demo.py [Trained File] # Demo face recognition

![demo1](./img1.jpg)
![demo2](./img2.jpg)
