# AthleteRise - AI-Powered Cricket Analytics

Complete cricket cover drive analysis system with real-time pose estimation, biomechanical analysis, and performance evaluation.

## Features

### Core Analysis
- ✅ Full video processing with pose estimation
- ✅ Biomechanical metrics calculation (elbow angle, spine lean, head-knee alignment)
- ✅ Live overlays with real-time feedback
- ✅ Multi-category performance scoring
- ✅ Comprehensive evaluation reports

### Advanced Features (BONUS)
- ✅ Automatic phase segmentation (Stance → Stride → Downswing → Impact → Follow-through → Recovery)
- ✅ Contact moment auto-detection
- ✅ Temporal smoothness & consistency analysis
- ✅ Real-time performance (10+ FPS target)
- ✅ Reference comparison with ideal technique
- ✅ Skill grade prediction (Beginner/Intermediate/Advanced)
- ✅ Streamlit web interface
- ✅ Comprehensive report generation

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd athleterise-cricket-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the analysis**

### Command Line Usage
```bash
python cover_drive_analysis_realtime.py --url "https://youtube.com/shorts/vSX3IRxGnNY"
```

### Streamlit Web Interface
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## Usage

### Command Line Options
```bash
# Analyze YouTube video
python cover_drive_analysis_realtime.py --url "https://youtube.com/shorts/vSX3IRxGnNY"

# Analyze local video file
python cover_drive_analysis_realtime.py --video "path/to/video.mp4"

# Use custom config
python cover_drive_analysis_realtime.py --config "custom_config.json"

# Enable maximum speed optimizations
python cover_drive_analysis_realtime.py --fast --url "https://youtube.com/shorts/vSX3IRxGnNY"
```

### Streamlit Interface
1. Open the web interface: `streamlit run app.py`
2. Upload a video file OR paste a YouTube URL
3. Click "Analyze Cricket Video"
4. View results and download analysis files

## Output Files

The system generates:

- `output/annotated_video.mp4` - Full video with pose overlays and metrics
- `output/evaluation.json` - Detailed analysis results in JSON format
- `output/evaluation.txt` - Human-readable summary report
- `output/smoothness_analysis.png` - Consistency charts (if enabled)

## Configuration

Edit `config.json` to customize:

```json
{
  "pose_detection": {
    "pose_confidence": 0.7,
    "pose_detection_confidence": 0.6
  },
  "analysis_thresholds": {
    "elbow_min": 90,
    "elbow_max": 140,
    "spine_lean_max": 15
  },
  "features": {
    "enable_phase_detection": true,
    "enable_contact_detection": true,
    "enable_smoothness_analysis": true
  }
}
```

## Performance

- **Target**: ≥10 FPS processing speed
- **Typical**: 8-15 FPS on modern CPU
- **Optimization**: Multiple speed/quality presets available

## Analysis Metrics

### Biomechanical Measurements
1. **Front Elbow Angle** - Shoulder-elbow-wrist angle
2. **Spine Lean** - Hip-shoulder line vs vertical
3. **Head-Knee Alignment** - Vertical distance between head and front knee
4. **Front Foot Direction** - Foot angle relative to crease
5. **Balance Score** - Hip alignment and stability

### Performance Categories
1. **Swing Control** (1-10) - Technique and consistency
2. **Balance** (1-10) - Stability throughout shot
3. **Head Position** (1-10) - Head positioning over front leg
4. **Footwork** (1-10) - Stance and movement quality
5. **Follow-through** (1-10) - Shot