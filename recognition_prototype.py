import cv2
import numpy as np
import face_recognition
from anti_spoofing import AntiSpoof, make_prediction
import time
import os
import csv
import sys 

# --- CONFIGURATION AND HYPERPARAMETERS ---
# IMPORTANT: These paths must be correctly set before running.
# The video and image must be downloaded locally to simulate edge deployment.
VIDEO_PATH = "./data/sample.mp4" 
REFERENCE_IMAGE_PATH = "./data/mark.jpg"
ANTI_SPOOF_MODEL_PATH = "./model/AntiSpoofing_bin_128.onnx"

# Key decision: Face Recognition Threshold (Euclidean Distance).
# This hyperparameter must be validated (Part B). A lower value means stricter matching.
FACE_DISTANCE_THRESHOLD = 0.60 
FACE_ANTI_SPOOF_THRESHOLD = 0.5

# Production optimization: Process only every Nth frame.
# FRAME_SKIP_N=1 for max accuracy/recall; FRAME_SKIP_N=5 for faster speed/lower CPU load.
FRAME_SKIP_N = 1

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

# --- Helper Function for Output ---
def save_results(results: list, output_filename="frame_results.csv"):
    """Saves the identified frame indices and metadata to a CSV file."""
    
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    output_path = os.path.join('results', output_filename)
    
    print(f"\n4. Saving results to {output_path}...")
    
    if not results:
        print("   No matching frames found. Saving empty file.")
        # Create an empty file for the deliverable
        with open(output_path, "w") as f:
            f.write("Frame Index,Timestamp (ms),Minimum Distance\n")
        return

    fieldnames = ['frame_index', 'timestamp_ms', 'min_distance']
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("   Results saved successfully.")


def check_liveness(anti_spoof, rgb_frame, frame_count, idx):
    face_locations = face_recognition.face_locations(rgb_frame)
    min_face_bbox = face_locations[idx]
    pred = make_prediction(rgb_frame, min_face_bbox, anti_spoof)
    
    if not pred: 
        return False

    (x1, y1, x2, y2), label, score = pred
    if label == 0:
        if score > FACE_ANTI_SPOOF_THRESHOLD:
            res_text = "REAL  {:.2f}".format(score)
            color = COLOR_REAL
        else: 
            res_text = "unknown"
            color = COLOR_UNKNOWN
    else:
        res_text = "FAKE      {:.2f}".format(score)
        color = COLOR_FAKE

    if "REAL" not in res_text:
        # draw bbox with label
        width, height = rgb_frame.shape[:2]

        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(rgb_frame, res_text, (x1, y1 - height//50), 
                    cv2.FONT_HERSHEY_COMPLEX, (x2-x1)//50, color, width//50)
        
        output_path = './fake_image'
        os.makedirs(output_path, exist_ok=True)
        filename = f"{frame_count}-{score:.3f}.jpg"
        save_path = os.path.join(output_path, filename)
    
        # 6. Save the image
        cv2.imwrite(save_path, rgb_frame)

        return False
    return True


# --- CORE LOGIC FUNCTIONS ---

def extract_reference_embedding(image_path: str) -> np.ndarray:
    """
    1. Extracts the 128D embedding for the reference person.
    """
    print(f"1. Processing reference image: {image_path}...")
    try:
        # Load image (OpenCV reads BGR by default)
        ref_image_bgr = cv2.imread(image_path)
        # Convert BGR to RGB (Dlib/face_recognition expects RGB)
        ref_image_rgb = cv2.cvtColor(ref_image_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        print(f"ERROR: Cannot read reference image at {image_path}. Ensure it is downloaded.")
        sys.exit(1)

    # CRITICAL FIX: Explicitly cast to np.uint8 to prevent Dlib/C++ TypeError
    ref_image_rgb = ref_image_rgb.astype(np.uint8)

    # Detect faces
    face_locations = face_recognition.face_locations(ref_image_rgb)

    if not face_locations:
        print("ERROR: No face detected in the reference image.")
        return None

    # Extract the embedding using the most robust API call (passing locations)
    reference_embeddings = face_recognition.face_encodings(ref_image_rgb, face_locations)
    
    if not reference_embeddings:
        print("ERROR: Could not extract face encoding.")
        return None
        
    print(f"   Reference face embedding extracted successfully.")
    return reference_embeddings[0]


def find_target_in_video(video_path: str, known_embedding: np.ndarray, 
                         anti_spoof: int, threshold: float, frame_skip_n: int):
    """
    2. & 3. Iterates through video frames, performs face recognition, and matches.
    Includes adjustable frame skipping and progress display.
    """
    print(f"\n2. Processing video: {video_path}...")
    print(f"   Performance setting: Processing every {frame_skip_n} frames.")
    
    frames_containing_person = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file {video_path}. Ensure it is downloaded.")
        sys.exit(1)

    # Get total frames for progress bar (CV_CAP_PROP_FRAME_COUNT)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"   Total frames in video: {total_frames}")

    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_count += 1
        
        # --- PROGRESS DISPLAY (Shows progress even on skipped frames) ---
        if frame_count % 50 == 0 or frame_count == total_frames:
            progress_percentage = (frame_count / total_frames) * 100
            sys.stdout.write(f"\r   Processing progress: Frame {frame_count}/{total_frames} ({progress_percentage:.1f}%)")
            sys.stdout.flush()
            
        # --- FRAME SKIPPING LOGIC ---
        if frame_count % frame_skip_n != 0:
            continue
        # ----------------------------
        
        # Performance: Resize frame for faster processing (e.g., 50% scale)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # CRITICAL FIX: Explicit data type cast for Dlib compatibility
        rgb_frame = rgb_frame.astype(np.uint8)

        # Feature Extraction: Detect and encode faces
        frame_embeddings_list = face_recognition.face_encodings(rgb_frame)
        
        if frame_embeddings_list:
            
            # 3. Similarity Matching (Euclidean Distance)
            
            # CRITICAL FIX: Convert list to NumPy array explicitly to avoid list/list TypeError
            frame_embeddings = np.array(frame_embeddings_list)
            
            # Calculate the Euclidean distance (L2 norm) using pure NumPy
            # This is equivalent to face_recognition.face_distance but avoids wrapper bugs.
            distances = np.linalg.norm(frame_embeddings - known_embedding, axis=1)

            # 4. Decision and Logging
            min_distance = np.min(distances)
            min_idx = np.argmin(distances)

            if min_distance > threshold or \
                not check_liveness(anti_spoof, rgb_frame, frame_count, min_idx):
                continue

            frames_containing_person.append({
                'frame_index': frame_count,
                'min_distance': float(min_distance),
                'timestamp_ms': cap.get(cv2.CAP_PROP_POS_MSEC)
            })
            # Print match status above the progress bar
            sys.stdout.write(f"\r{' ' * 80}\r") # Clear the progress line
            print(f"   MATCH FOUND: Frame {frame_count} (Distance: {min_distance:.3f})")
            sys.stdout.flush()
            
    cap.release()
    
    # Final cleanup of the progress bar line
    sys.stdout.write(f"\r{' ' * 80}\r") 
    
    total_time = time.time() - start_time
    # total_frames_read is total_frames here (approximate)
    total_frames_processed = frame_count // frame_skip_n

    print(f"\nProcessing complete.")
    print(f"Total frames read: {total_frames}. Frames processed: {total_frames_processed}. Total time: {total_time:.2f} seconds.")
    
    return frames_containing_person

if __name__ == "__main__":
    
    print("--- Technical Challenge: Senior MLE (Computer Vision) Part A ---")
    print(f"Constraint: Edge Deployment Simulation (Offline)")
    print(f"Disclosed AI Tools: OpenCV, dlib (via face_recognition), NumPy.")
    print(f"Core Assumptions: Threshold={FACE_DISTANCE_THRESHOLD}, FrameSkip={FRAME_SKIP_N}.")
    print("---------------------------------------------------------------")
    
    # Step 1: Extract reference embedding
    reference_embedding = extract_reference_embedding(REFERENCE_IMAGE_PATH)
    anti_spoof = AntiSpoof(ANTI_SPOOF_MODEL_PATH)
    
    if reference_embedding is not None:
        # Step 2 & 3: Process video and match
        match_results = find_target_in_video(
            VIDEO_PATH, 
            reference_embedding, 
            anti_spoof,
            FACE_DISTANCE_THRESHOLD,
            FRAME_SKIP_N 
        )
        
        # Step 4: Deliver output
        save_results(match_results)