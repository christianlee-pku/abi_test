### Setup Instructions for Tech Challenge: Senior MLE (Computer Vision)

This document provides instructions to set up the environment and run the `recognition_prototype.py` script locally, including the integrated Anti-Spoofing (Liveness Detection) module.

#### Prerequisites

1.  **Python:** Ensure you have Python 3.8+ installed.
2.  **Git:** Required for cloning the repository.
3.  **Local Storage:** Sufficient disk space for the video and model files.

#### Step 1: Clone Repository and Create Environment

```bash
# Clone this private repository
git clone [YOUR_REPOSITORY_URL]
cd TechChallenge_SeniorMLE

# Create and activate a dedicated virtual environment
conda create -n recog_env python=3.10
conda activate recog_env
# OR
# python -m venv recog_env
# source recog_env/bin/activate
```

#### Step 2: Install Dependencies
The system requires libraries for core CV, face recognition, and ONNX model inference.

```bash
# Install core dependencies (numpy, opencv, dlib-based face-recognition)
pip install numpy opencv-python face-recognition

# Install ONNX Runtime for the Anti-Spoofing Model (.onnx)
# This is crucial for fast, edge-optimized model inference.
pip install onnxruntime
```

#### Step 3: Setup Directory Structure and Download Files
You must create the necessary directories and download all required files into the corresponding paths as specified in recognition_prototype.py

# Create necessary subdirectories
mkdir data
mkdir model

# 1. Download Video: 
#    Download the YouTube video (JWYPT4r9kLU) and save it as sample.mp4.
#    Source Link: [https://www.youtube.com/watch?v=JWYPT4r9kLU](https://www.youtube.com/watch?v=JWYPT4r9kLU)
#    Move file: mv your_downloaded_video.mp4 ./data/sample.mp4

# 2. Download Reference Image: 
#    Download the reference image (Mark Ruffalo) and save it as mark.jpg.
#    Source Link: [https://www.britannica.com/biography/Mark-Ruffalo](https://www.britannica.com/biography/Mark-Ruffalo)
#    Move file: mv your_downloaded_image.jpg ./data/mark.jpg

# 3. Anti-Spoofing Model:
#    Download the ONNX model weights and save it to the model directory.
#    REQUIRED FILE: AntiSpoofing_bin_128.onnx
#    Move file: mv your_downloaded_model.onnx ./model/AntiSpoofing_bin_128.onnx

# 4. Anti-Spoofing Code:
#    You must place the external Python file containing the AntiSpoof and make_prediction logic 
#    in the root directory.
#    REQUIRED FILE: anti_spoofing.py


#### Step 4: Run the Prototype
Execute the main script. The output will display real-time progress, matches found, and total processing time. Frames failing the liveness check will be saved to the ./fake_image directory.

```bash
python recognition_prototype.py
```

#### Step 5: Check Results

```bash
# View the final results CSV file (frames passing BOTH recognition and liveness)
cat results/frame_results.csv

# Check the directory containing images that FAILED the liveness check
ls fake_image/
```