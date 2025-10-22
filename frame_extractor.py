import os
import csv
import cv2
import sys

# --- Configuration Constants (Must match settings in recognition_prototype.py) ---
# The paths must be set to where your video and results file are located.
VIDEO_PATH = "./data/sample.mp4" 
OUTPUT_RESULTS_FILENAME = "frame_results.csv"
OUTPUT_FRAMES_DIR = "matched_frames"
FRAME_SKIP_N = 5
# --------------------------------------------------------------------------------    

def extract_and_save_matched_frames(video_path: str, results_filename: str, output_dir: str):
    """
    Reads the CSV results file, extracts corresponding frames from the video based on 
    frame index and minimum distance, and saves them with the format: "[frame_index]-[min_distance].jpg".
    
    Args:
        video_path (str): Path to the source video file.
        results_filename (str): Name of the CSV file containing match data.
        output_dir (str): Directory where matched frame images will be saved.
    """
    
    # 1. Set paths
    # Assume results_part_a.csv is located in the 'results/' directory
    results_path = os.path.join('results', results_filename)
    output_path = os.path.join(output_dir)
    
    if not os.path.exists(results_path):
        print(f"ERROR: Results file not found at {results_path}. Please run recognition_prototype.py first.")
        return

    # Create the output directory
    os.makedirs(output_path, exist_ok=True)
    print(f"\n--- Matched Frame Extraction Tool ---")
    print(f"Reading results file: {results_path}")
    print(f"Extracting and saving matched frames to directory: {output_path}")

    # 2. Read CSV file content
    matched_frames_data = []
    try:
        with open(results_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Attempt to safely convert data types
                try:
                    matched_frames_data.append({
                        'frame_index': int(row['frame_index']),
                        'min_distance': float(row['min_distance'])
                    })
                except (ValueError, KeyError) as e:
                    print(f"WARNING: Skipping row with incorrect format or missing keys in CSV: {row}. Error: {e}")
                    continue
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not matched_frames_data:
        print("No matching frame data found in CSV. Operation skipped.")
        return
    
    total_matches = len(matched_frames_data)
    
    # 3. Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FATAL ERROR: Could not open video file {video_path}. Please check path and filename.")
        return

    # 4. Iterate and save each matched frame
    for i, data in enumerate(matched_frames_data):
        if i % FRAME_SKIP_N :
            continue
        frame_index = data['frame_index']
        min_distance = data['min_distance']
        
        # --- Progress Display ---
        sys.stdout.write(f"\rProcessing progress: {i+1}/{total_matches} frames (Frame: {frame_index})")
        sys.stdout.flush()
        
        # Set video read position to the target frame
        # CV_CAP_PROP_POS_FRAMES is 0-based, so we must subtract 1 from the 1-based frame_index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        
        ret, frame = cap.read()
        
        if ret:
            # 5. Construct filename: [frame_index]-[min_distance].jpg
            # Format min_distance to 3 decimal places and replace the decimal point with an underscore for safety
            # Example: 0.54321 -> 0.543
            
            # Filename format: frame_index-min_distance.jpg
            filename = f"{frame_index}-{min_distance:.3f}.jpg"
            save_path = os.path.join(output_path, filename)
            
            # 6. Save the image
            cv2.imwrite(save_path, frame)
        else:
            print(f"\nWARNING: Could not read frame {frame_index} from the video.")

    cap.release()
    
    sys.stdout.write(f"\r{' ' * 80}\r") # Clear the progress bar
    print(f"\nExtraction complete. Successfully saved {total_matches} matched frames to the {output_dir}/ directory.")


if __name__ == "__main__":
    # Ensure recognition_prototype.py has been run successfully and results/results_part_a.csv exists
    extract_and_save_matched_frames(
        VIDEO_PATH, 
        OUTPUT_RESULTS_FILENAME, 
        OUTPUT_FRAMES_DIR
    )