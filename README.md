# Player-tracking-in-and-out-of-frame-with-yolov11
Requirements
text
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
filterpy>=1.4.5
torch>=1.8.0
torchvision>=0.9.0
 Installation & Setup
Google Colab Setup
Install Dependencies

python
!pip install ultralytics opencv-python-headless numpy pandas filterpy scikit-learn
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Upload Required Files

python
from google.colab import files

# Upload your fine-tuned model
print("Upload your best.pt model file:")
uploaded_model = files.upload()

# Upload your video file
print("Upload your video file:")
uploaded_video = files.upload()
Set File Paths

python
# If files are already uploaded to Colab
model_path = "best.pt"  # Your model filename
video_path = "your_video.mp4"  # Your video filename

# Verify files exist
import os
assert os.path.exists(model_path), f"Model file not found: {model_path}"
assert os.path.exists(video_path), f"Video file not found: {video_path}"
Local Environment Setup
Clone Repository

bash
git clone <repository-url>
cd player-tracking-system
Create Virtual Environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

bash
pip install -r requirements.txt


Usage
Basic Usage
python
from player_tracker import AdvancedPlayerTracker

# Initialize tracker
tracker = AdvancedPlayerTracker(
    model_path="best.pt",
    confidence_threshold=0.5,
    iou_threshold=0.7
)

# Process video
report, frames = tracker.process_video(
    video_path="input_video.mp4",
    output_path="tracked_output.mp4"
)

# Generate reports
tracker.generate_reports(report)
Advanced Configuration
python
# Custom tracker settings
tracker = AdvancedPlayerTracker(
    model_path="best.pt",
    confidence_threshold=0.6,  # Higher confidence threshold
    iou_threshold=0.8,         # Stricter IoU matching
    max_disappeared=45,        # Frames before player considered lost
    min_track_length=8         # Minimum track length for validation
)