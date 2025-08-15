import streamlit as st
import os
import time
import json
from pathlib import Path
import tempfile
import cv2
import base64
from PIL import Image
import numpy as np

# FIXED: Proper import with error handling
try:
    from cover_drive_analysis_realtime import CricketPoseAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError as e:
    st.error(f"Cricket analyzer not available: {e}")
    st.info("Make sure cover_drive_analysis_realtime.py is in the same directory")
    ANALYZER_AVAILABLE = False

# Create wrapper function for compatibility
def analyze_video(video_path, progress_callback=None):
    """Analyze video using the cricket pose analyzer"""
    if not ANALYZER_AVAILABLE:
        return {}
    
    try:
        analyzer = CricketPoseAnalyzer()
        analyzer.config["input_path"] = video_path
        
        success = analyzer.process_video(video_path, progress_callback)
        if success:
            return analyzer.generate_evaluation_report()
        else:
            return {}
    except Exception as e:
        st.error(f"Analysis error: {e}")
        return {}

# Set page configuration
st.set_page_config(
    page_title="AthleteRise - Cricket Analysis Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# COMPLETE CSS WITH ALL FIXES
def load_css():
    st.markdown("""
    <style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    /* Global Styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Professional Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #1e40af 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main-header h1 {
        font-family: 'Manrope', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: none;
        letter-spacing: -0.5px;
    }

    .main-header .subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    .logo {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 800;
        font-size: 2.2rem;
    }

    .logo-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(45deg, #60a5fa, #3b82f6);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.2rem;
    }

    /* Section Headers */
    .section-header {
        font-family: 'Manrope', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* FIXED: File Upload Styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e0 !important;
        border-radius: 12px !important;
        background: linear-gradient(145deg, #f8fafc, #f1f5f9) !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        min-height: 120px !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6 !important;
        background: linear-gradient(145deg, #eff6ff, #dbeafe) !important;
        transform: translateY(-1px) !important;
    }

    /* CRITICAL FIX: Video Display - Force proper sizing and visibility */
    [data-testid="stVideo"] {
        margin: 1rem 0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        background: #000 !important;
        min-height: 200px !important;
        width: 100% !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: relative !important;
    }

    [data-testid="stVideo"] video {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        min-height: 200px !important;
        border-radius: 8px !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        object-fit: contain !important;
        background: #000 !important;
        z-index: 1 !important;
    }

    /* Force video container to be visible */
    .stVideo > div {
        width: 100% !important;
        height: auto !important;
        display: block !important;
        visibility: visible !important;
        position: relative !important;
    }

    /* Additional video fix - target specific video elements */
    video {
        width: 100% !important;
        height: auto !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Professional Metric Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(30, 58, 138, 0.12);
    }

    [data-testid="metric-container"] label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-family: 'Manrope', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e3a8a !important;
    }

    /* Score bars */
    .score-bar {
        width: 100%;
        height: 4px;
        background-color: #e2e8f0;
        border-radius: 2px;
        margin-top: 8px;
        overflow: hidden;
    }

    .score-fill {
        height: 100%;
        border-radius: 2px;
        transition: width 0.8s ease;
    }

    .score-excellent { background: linear-gradient(90deg, #10b981, #059669); }
    .score-good { background: linear-gradient(90deg, #3b82f6, #1d4ed8); }
    .score-fair { background: linear-gradient(90deg, #f59e0b, #d97706); }
    .score-poor { background: linear-gradient(90deg, #ef4444, #dc2626); }

    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.8rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.3) !important;
        letter-spacing: 0.3px !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4) !important;
    }

    /* Professional Alerts */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0) !important;
        color: #047857 !important;
        border: 1px solid #a7f3d0 !important;
        border-radius: 8px !important;
    }

    .stError {
        background: linear-gradient(135deg, #fee2e2, #fca5a5) !important;
        color: #b91c1c !important;
        border: 1px solid #fca5a5 !important;
        border-radius: 8px !important;
    }

    .stWarning {
        background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
        color: #92400e !important;
        border: 1px solid #fde68a !important;
        border-radius: 8px !important;
    }

    .stInfo {
        background: linear-gradient(135deg, #dbeafe, #93c5fd) !important;
        color: #1d4ed8 !important;
        border: 1px solid #93c5fd !important;
        border-radius: 8px !important;
    }

    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        margin: 0.2rem 0 !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
    """, unsafe_allow_html=True)

def app_header():
    st.markdown("""
    <div class="main-header">
        <div class="logo">
            <div class="logo-icon">A</div>
            AthleteRise
        </div>
        <div class="subtitle">AI-Powered Cricket Performance Analysis</div>
    </div>
    """, unsafe_allow_html=True)

def process_and_analyze_video(uploaded_file, youtube_url, analysis_mode, confidence_threshold, frame_sampling):
    """Process and analyze video with enhanced error handling"""
    
    if not ANALYZER_AVAILABLE:
        st.error("Cricket analyzer not available. Please check the installation.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    temp_file = None
    video_path = None
    
    try:
        # Step 1: Prepare video
        status_text.text("🔄 Preparing video...")
        progress_bar.progress(10)
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        if uploaded_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file.flush()
            video_path = temp_file.name
            status_text.text("✅ Video file uploaded successfully")
            
        elif youtube_url:
            status_text.text("📥 Downloading video from YouTube...")
            progress_bar.progress(20)
            
            try:
                analyzer = CricketPoseAnalyzer()
                analyzer.config["output_dir"] = str(output_dir)
                video_path = analyzer.download_video(youtube_url)
                
                if not video_path or not os.path.exists(video_path):
                    st.error("❌ Failed to download video. Please check the URL and try again.")
                    return
                    
                status_text.text("✅ Video downloaded successfully")
                
            except Exception as e:
                st.error(f"❌ YouTube download error: {str(e)}")
                return
        
        progress_bar.progress(40)
        status_text.text("⚙️ Starting comprehensive analysis...")
        
        # Step 2: Full analysis
        analyzer = CricketPoseAnalyzer()
        analyzer.config["output_dir"] = str(output_dir)
        
        def progress_callback(percent):
            progress_bar.progress(min(90, 40 + int(percent * 0.5)))
            status_text.text(f"🤖 Analyzing cricket technique... {percent:.1f}%")
        
        success = analyzer.process_video(video_path, progress_callback)
        
        if not success:
            st.error("❌ Video analysis failed. Please try with a different video.")
            return
        
        progress_bar.progress(95)
        status_text.text("📊 Generating detailed evaluation...")
        
        # Step 3: Generate comprehensive evaluation
        evaluation = analyzer.generate_evaluation_report()
        
        progress_bar.progress(100)
        status_text.text("✅ Complete analysis finished!")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Store results
        st.session_state['analysis_results'] = evaluation
        st.session_state['analyzer'] = analyzer
        st.session_state['show_results'] = True
        
        st.success("🎉 Comprehensive cricket analysis completed successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")
        
    finally:
        if temp_file:
            try:
                temp_file.close()
                os.unlink(temp_file.name)
            except:
                pass

# MAIN FIX: Complete updated display function
def display_analysis_results(evaluation, analyzer):
    """Display comprehensive analysis results with fixed video display"""
    
    st.markdown('<h2 class="section-header">Cricket Analysis Results</h2>', unsafe_allow_html=True)
    
    # Summary section
    summary = evaluation.get('summary', {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Skill Grade", summary.get('skill_grade', 'N/A'))
    with col2:
        st.metric("Processing FPS", f"{summary.get('average_processing_fps', 0):.1f}")
    with col3:
        st.metric("Phases Detected", summary.get('phase_transitions', 0))
    with col4:
        contact_frame = summary.get('contact_frame_detected')
        st.metric("Contact Detected", "Yes" if contact_frame else "No")
    
    # Performance scores
    st.subheader("Performance Scores")
    scores = evaluation.get('scores', {})
    
    if scores:
        score_cols = st.columns(len(scores))
        
        for i, (category, score) in enumerate(scores.items()):
            with score_cols[i]:
                display_name = category.replace('_', ' ').title()
                st.metric(
                    label=display_name,
                    value=f"{score}/10"
                )
                
                # Score bar
                score_percentage = (score / 10) * 100
                if score >= 8:
                    bar_class = "score-excellent"
                elif score >= 6:
                    bar_class = "score-good" 
                elif score >= 4:
                    bar_class = "score-fair"
                else:
                    bar_class = "score-poor"
                
                st.markdown(f"""
                <div class="score-bar">
                    <div class="score-fill {bar_class}" style="width: {score_percentage}%;"></div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # FIXED VIDEO DISPLAY SECTION
    video_col, analysis_col = st.columns([3, 2])
    
    with video_col:
        st.subheader("Annotated Analysis Video")
        
        output_dir = Path(analyzer.config.get("output_dir", "output"))
        
        # Look for video files in order of preference
        video_files = [
            "annotated_video.mp4",
            "annotated_video.avi", 
            "output_video.mp4",
            "cricket_analysis.mp4",
            "result.mp4"
        ]
        
        video_path = None
        
        # Find the first existing video file with reasonable size
        for filename in video_files:
            potential_path = output_dir / filename
            if potential_path.exists():
                file_size = potential_path.stat().st_size
                if file_size > 1024:  # At least 1KB
                    video_path = potential_path
                    break
        
        if video_path and video_path.exists():
            try:
                # Display video file info
                file_size_mb = video_path.stat().st_size / (1024 * 1024)
                st.info(f"📹 Video found: {video_path.name} ({file_size_mb:.1f} MB)")
                
                # Verify video is readable with OpenCV
                cap = cv2.VideoCapture(str(video_path))
                is_valid = cap.isOpened()
                
                if is_valid:
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps if fps > 0 else 0
                    
                    cap.release()
                    
                    # Display video properties
                    st.caption(f"Resolution: {width}×{height} | Duration: {duration:.1f}s | FPS: {fps:.1f}")
                    
                    # Primary display method: Read video as bytes
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                    
                    # Display using Streamlit's video component
                    st.video(video_bytes)
                    
                    # Success message
                    st.success("✅ Analysis video loaded successfully!")
                    
                else:
                    cap.release()
                    st.error("❌ Video file exists but cannot be read. File may be corrupted.")
                    
            except Exception as e:
                st.error(f"❌ Error loading video: {str(e)}")
                st.info("The video file exists but cannot be displayed. You can still download it below.")
            
            # Always provide download option
            try:
                with open(video_path, "rb") as f:
                    video_data = f.read()
                st.download_button(
                    "📥 Download Analysis Video",
                    data=video_data,
                    file_name=f"cricket_analysis_{int(time.time())}.mp4",
                    mime="video/mp4",
                    help="Download the annotated video to view locally"
                )
            except Exception as e:
                st.error(f"Download error: {e}")
                
        else:
            # No video found - show debug info
            st.warning("❌ No analysis video found")
            
            if output_dir.exists():
                files = list(output_dir.glob("*"))
                if files:
                    st.info("📂 Files in output directory:")
                    for file in files:
                        try:
                            size_kb = file.stat().st_size / 1024
                            st.write(f"• {file.name} ({size_kb:.1f} KB)")
                        except:
                            st.write(f"• {file.name} (unknown size)")
                else:
                    st.info("📂 Output directory exists but is empty")
            else:
                st.error("📂 Output directory not found")
            
            # Offer manual path input for debugging
            with st.expander("🔧 Debug: Manual Video Path"):
                manual_path = st.text_input(
                    "Enter video file path:",
                    placeholder="/path/to/your/video.mp4"
                )
                
                if manual_path and st.button("Try Manual Path"):
                    manual_video_path = Path(manual_path)
                    if manual_video_path.exists():
                        try:
                            with open(manual_video_path, 'rb') as f:
                                manual_video_bytes = f.read()
                            st.video(manual_video_bytes)
                            st.success("✅ Manual path video loaded!")
                        except Exception as e:
                            st.error(f"Manual path error: {e}")
                    else:
                        st.error("Manual path does not exist")
    
    # Analysis column - detailed feedback
    with analysis_col:
        st.subheader("Detailed Feedback")
        
        feedback = evaluation.get('feedback', {})
        if feedback:
            for category, comments in feedback.items():
                category_name = category.replace('_', ' ').title()
                
                with st.expander(category_name, expanded=True):
                    for comment in comments:
                        if any(word in comment.lower() for word in ['good', 'excellent', 'great', 'well']):
                            st.success(f"✅ {comment}")
                        elif any(word in comment.lower() for word in ['work on', 'improve', 'focus', 'needs']):
                            st.warning(f"⚠️ {comment}")
                        else:
                            st.info(f"ℹ️ {comment}")
        else:
            st.info("No detailed feedback available")
        
        # Phase Analysis
        phase_data = evaluation.get('phase_analysis', {})
        if phase_data.get('phases_detected'):
            st.subheader("Phase Analysis")
            phases = phase_data['phases_detected']
            st.write("**Phases Detected:**", " → ".join(phases))
            
            if phase_data.get('phase_timings'):
                st.write("**Timing:**")
                for phase, timing in phase_data['phase_timings'].items():
                    st.write(f"• {phase}: {timing}")
        
        # Reference Comparison
        ref_comp = evaluation.get('reference_comparison', {})
        if ref_comp:
            st.subheader("Technique Comparison")
            for metric, deviation in ref_comp.items():
                metric_name = metric.replace('_', ' ').title()
                if abs(deviation) < 5:
                    st.success(f"✅ {metric_name}: {deviation:+.1f}° (Good)")
                elif abs(deviation) < 10:
                    st.warning(f"⚠️ {metric_name}: {deviation:+.1f}° from ideal")
                else:
                    st.error(f"❌ {metric_name}: {deviation:+.1f}° (Needs work)")
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON Report
        if evaluation:
            report_json = json.dumps(evaluation, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON Report",
                data=report_json,
                file_name=f"cricket_analysis_report_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
                help="Complete analysis data in JSON format"
            )
    
    with col2:
        # Video Download
        if video_path and video_path.exists():
            try:
                with open(video_path, "rb") as f:
                    video_data = f.read()
                st.download_button(
                    label="🎬 Download Video",
                    data=video_data,
                    file_name=f"cricket_analysis_video_{int(time.time())}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                    help="Annotated analysis video"
                )
            except Exception as e:
                st.button(
                    "🎬 Video Unavailable",
                    disabled=True,
                    use_container_width=True,
                    help=f"Error: {e}"
                )
        else:
            st.button(
                "🎬 No Video Found",
                disabled=True,
                use_container_width=True,
                help="Analysis video was not generated"
            )
    
    with col3:
        # Text Summary
        text_report_path = output_dir / "evaluation.txt"
        if text_report_path.exists():
            try:
                with open(text_report_path, "r", encoding='utf-8') as f:
                    text_data = f.read()
                st.download_button(
                    label="📝 Download Summary",
                    data=text_data,
                    file_name=f"cricket_summary_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Text summary of analysis"
                )
            except Exception as e:
                st.button(
                    "📝 Summary Unavailable", 
                    disabled=True,
                    use_container_width=True,
                    help=f"Error: {e}"
                )
        else:
            st.button(
                "📝 No Summary Found",
                disabled=True, 
                use_container_width=True,
                help="Text summary was not generated"
            )


def ensure_video_compatibility(video_path, output_path):
    """
    Ensure video is compatible with Streamlit display
    Call this after video processing in your analyzer
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        # Get properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Use web-compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'H264'
        
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
            isColor=True
        )
        
        if not out.isOpened():
            cap.release()
            return False
        
        # Copy frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
        return Path(output_path).exists()
        
    except Exception as e:
        print(f"Video compatibility error: {e}")
        return False

def ensure_web_compatible_video(input_path, output_path):
    """
    Ensure video is web-compatible for Streamlit display
    Add this to your CricketPoseAnalyzer class
    """
    import cv2
    
    try:
        # Read the input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open input video {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define codec - use H.264 for web compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Alternative codec options if mp4v doesn't work:
        # fourcc = cv2.VideoWriter_fourcc(*'H264')  # H.264
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Xvid
        
        # Create video writer with web-compatible settings
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps, 
            (width, height),
            isColor=True
        )
        
        if not out.isOpened():
            print("Error: Could not open output video writer")
            cap.release()
            return False
        
        print(f"Converting video for web compatibility...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Settings: {width}x{height}, {fps}fps")
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Verify the output file
        if Path(output_path).exists() and Path(output_path).stat().st_size > 1024:
            print(f"✅ Web-compatible video created: {output_path}")
            return True
        else:
            print(f"❌ Failed to create web-compatible video")
            return False
            
    except Exception as e:
        print(f"Error in video conversion: {e}")
        return False

# Usage in your process_video method:
def process_video_with_web_output(self, video_path, progress_callback=None):
    """
    Modified process_video method that ensures web-compatible output
    Add this to your CricketPoseAnalyzer class
    """
    
    # ... your existing processing code ...
    
    # After creating the annotated video, ensure it's web-compatible
    original_output = Path(self.config["output_dir"]) / "annotated_video.mp4"
    web_compatible_output = Path(self.config["output_dir"]) / "web_compatible_video.mp4"
    
    if original_output.exists():
        print("Creating web-compatible version...")
        if self.ensure_web_compatible_video(original_output, web_compatible_output):
            # Replace original with web-compatible version
            import shutil
            shutil.move(web_compatible_output, original_output)
            print("✅ Video is now web-compatible")
        else:
            print("⚠️ Could not create web-compatible version, using original")
    
    return True
def video_processing_section():
    
    st.markdown('<h2 class="section-header">Video Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv"],
            help="Upload cricket batting video (MP4, MOV, AVI, MKV)"
        )
        
        st.subheader("Or Use YouTube URL")
        youtube_url = st.text_input(
            "YouTube video URL",
            placeholder="https://youtube.com/shorts/vSX3IRxGnNY",
            help="Paste a YouTube video URL"
        )
    
    with col2:
        st.subheader("Analysis Settings")
        
        analysis_mode = st.selectbox(
            "Processing Quality",
            ["Fast Processing", "Balanced (Recommended)", "High Accuracy"],
            index=1
        )
        
        confidence_threshold = st.slider("Pose Detection Confidence", 0.5, 0.9, 0.7)
        frame_sampling = st.slider("Frame Sampling Rate", 1, 5, 3)
        
        st.subheader("Start Analysis")
        analyze_button = st.button("🏏 Analyze Cricket Video", use_container_width=True, type="primary")
        
        if analyze_button:
            if uploaded_file or youtube_url:
                process_and_analyze_video(uploaded_file, youtube_url, analysis_mode, confidence_threshold, frame_sampling)
            else:
                st.error("❌ Please upload a video file or provide a YouTube URL")

def main():
    """Main Streamlit application"""
    # Initialize session state
    for key in ['analysis_results', 'analyzer', 'show_results']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'show_results' else False
    
    load_css()
    app_header()
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### About AthleteRise")
        st.info("Complete cricket analysis with pose detection, biomechanical metrics, and phase analysis.")
        
        st.markdown("### Key Features")
        st.markdown("""
        - Real-time pose estimation
        - Biomechanical analysis
        - Phase detection
        - Contact moment detection
        - Performance scoring
        - Detailed feedback
        """)
        
        st.markdown("### Troubleshooting")
        st.markdown("""
        **Video not showing?**
        - Check file format (MP4 recommended)
        - Ensure video file exists in output folder
        - Try the manual path option in debug section
        - Download video if display fails
        """)
    
    video_processing_section()
    
    # Show results
    if (st.session_state.get('show_results') and 
        st.session_state.get('analysis_results') and 
        st.session_state.get('analyzer')):
        
        st.markdown("---")
        display_analysis_results(
            st.session_state['analysis_results'], 
            st.session_state['analyzer']
        )
        
        if st.button("🔄 Analyze New Video"):
            for key in ['analysis_results', 'analyzer', 'show_results']:
                st.session_state[key] = None if key != 'show_results' else False
            st.rerun()

if __name__ == "__main__":
    main()