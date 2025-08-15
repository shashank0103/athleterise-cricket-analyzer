#!/usr/bin/env python3
"""
AthleteRise - Real-Time Cricket Cover Drive Analysis
Complete implementation based on assignment requirements
"""

import cv2
import numpy as np
import json
import time
import os
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import tempfile
import re
import math
from collections import deque
import matplotlib.pyplot as plt

# Import MediaPipe with error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("Error: MediaPipe not available. Install with: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

# Import yt-dlp with error handling
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("Warning: yt-dlp not available. Install with: pip install yt-dlp")

class CricketPoseAnalyzer:
    """Complete Cricket Pose Analyzer implementing all assignment requirements"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the cricket pose analyzer"""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")
            
        self.load_config(config_path)
        self.setup_mediapipe()
        self.setup_output_dir()
        
        # Metrics storage
        self.frame_metrics = []
        self.phase_data = []
        self.fps_tracker = deque(maxlen=30)
        self.contact_frame = None
        self.previous_landmarks = {}
        
        # Performance tracking
        self.processing_times = []
        
        # Phase detection variables
        self.phases = ['stance', 'stride', 'downswing', 'impact', 'follow_through', 'recovery']
        self.current_phase = 'stance'
        self.phase_transitions = []
        
    def load_config(self, config_path: str):
        """Load configuration with comprehensive defaults"""
        default_config = {
            "video_url": "https://youtube.com/shorts/vSX3IRxGnNY",
            "input_path": "input/cricket_video.mp4",
            "output_dir": "output",
            "pose_confidence": 0.7,
            "pose_detection_confidence": 0.6,
            "angle_thresholds": {
                "elbow_min": 90,
                "elbow_max": 140,
                "spine_lean_max": 15,
                "head_knee_max": 30,
                "foot_angle_optimal": 45
            },
            "fps_target": 15,
            "model_complexity": 1,
            "enable_phase_detection": True,
            "enable_contact_detection": True,
            "enable_smoothness_analysis": True,
            "target_fps": 10.0,
            "reference_comparison": {
                "elbow_ideal": 115,
                "spine_lean_ideal": 5,
                "head_knee_ideal": 10
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        self.config = default_config
        
    def setup_mediapipe(self):
        """Initialize MediaPipe pose estimation"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.config.get("model_complexity", 1),
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=self.config["pose_detection_confidence"],
            min_tracking_confidence=self.config["pose_confidence"]
        )
        
    def setup_output_dir(self):
        """Create output directory structure"""
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        Path("input").mkdir(parents=True, exist_ok=True)
        
    def download_video(self, url: str) -> Optional[str]:
        """Download video from YouTube using yt-dlp"""
        if not YT_DLP_AVAILABLE:
            print("Error: yt-dlp not available. Install with: pip install yt-dlp")
            return None
            
        print(f"Downloading video from: {url}")
        
        video_id = self.extract_video_id(url)
        output_filename = f"input/youtube_video_{video_id or int(time.time())}.%(ext)s"
        
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
            'outtmpl': output_filename,
            'quiet': False,
            'no_warnings': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
                # Find downloaded file
                downloaded_file = f"input/youtube_video_{video_id}.mp4" if video_id else None
                if not downloaded_file or not os.path.exists(downloaded_file):
                    # Find most recent file in input directory
                    input_dir = Path("input")
                    video_files = list(input_dir.glob("youtube_video_*"))
                    if video_files:
                        downloaded_file = str(max(video_files, key=lambda x: x.stat().st_ctime))
                
                if downloaded_file and os.path.exists(downloaded_file):
                    print(f"Video downloaded: {downloaded_file}")
                    return downloaded_file
                    
        except Exception as e:
            print(f"Download error: {e}")
            
        return None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=)([\w-]+)',
            r'(?:youtu\.be/)([\w-]+)',
            r'(?:youtube\.com/shorts/)([\w-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        try:
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            
            return angle
        except:
            return 0.0
    
    def extract_landmarks(self, results) -> Dict:
        """Extract key landmarks from MediaPipe results"""
        landmarks = {}
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # Define key landmarks for cricket analysis
            landmark_indices = {
                'nose': 0,
                'left_shoulder': 11, 'right_shoulder': 12,
                'left_elbow': 13, 'right_elbow': 14,
                'left_wrist': 15, 'right_wrist': 16,
                'left_hip': 23, 'right_hip': 24,
                'left_knee': 25, 'right_knee': 26,
                'left_ankle': 27, 'right_ankle': 28
            }
            
            for name, idx in landmark_indices.items():
                if idx < len(lm):
                    landmarks[name] = (lm[idx].x, lm[idx].y, lm[idx].visibility)
        
        return landmarks
    
    def calculate_biomechanical_metrics(self, landmarks: Dict, frame_width: int, frame_height: int) -> Dict:
        """Calculate comprehensive biomechanical metrics"""
        metrics = {}
        
        if not landmarks:
            return metrics
            
        try:
            # 1. Front elbow angle (assuming right-handed batsman)
            if all(k in landmarks for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                elbow_angle = self.calculate_angle(
                    landmarks['right_shoulder'][:2],
                    landmarks['right_elbow'][:2],
                    landmarks['right_wrist'][:2]
                )
                metrics['front_elbow_angle'] = elbow_angle
                
            # 2. Spine lean (hip-shoulder line vs vertical)
            if all(k in landmarks for k in ['right_shoulder', 'right_hip']):
                shoulder_y = landmarks['right_shoulder'][1]
                hip_y = landmarks['right_hip'][1]
                shoulder_x = landmarks['right_shoulder'][0]
                hip_x = landmarks['right_hip'][0]
                
                spine_angle = math.degrees(math.atan2(abs(shoulder_x - hip_x), abs(hip_y - shoulder_y)))
                metrics['spine_lean'] = spine_angle
                
            # 3. Head-over-knee alignment
            if all(k in landmarks for k in ['nose', 'right_knee']):
                head_x = landmarks['nose'][0] * frame_width
                knee_x = landmarks['right_knee'][0] * frame_width
                head_knee_distance = abs(head_x - knee_x)
                metrics['head_knee_alignment'] = head_knee_distance
                
            # 4. Front foot direction (approximate)
            if all(k in landmarks for k in ['right_knee', 'right_ankle']):
                knee_pos = landmarks['right_knee'][:2]
                ankle_pos = landmarks['right_ankle'][:2]
                foot_angle = math.degrees(math.atan2(ankle_pos[1] - knee_pos[1], ankle_pos[0] - knee_pos[0]))
                metrics['front_foot_angle'] = abs(foot_angle)
                
            # 5. Balance metric (hip alignment)
            if all(k in landmarks for k in ['left_hip', 'right_hip']):
                left_hip_y = landmarks['left_hip'][1]
                right_hip_y = landmarks['right_hip'][1]
                balance_score = abs(left_hip_y - right_hip_y) * 100
                metrics['balance'] = balance_score
                
            # 6. Joint velocities for phase detection
            if hasattr(self, 'previous_landmarks') and self.previous_landmarks:
                velocities = {}
                for joint in ['right_wrist', 'right_elbow']:
                    if joint in landmarks and joint in self.previous_landmarks:
                        prev_pos = self.previous_landmarks[joint][:2]
                        curr_pos = landmarks[joint][:2]
                        velocity = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                        velocities[f'{joint}_velocity'] = velocity
                metrics.update(velocities)
            
            self.previous_landmarks = landmarks.copy()
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            
        return metrics
    
    def detect_phase(self, metrics: Dict, frame_num: int) -> str:
        """Automatic phase detection using joint velocities and angles"""
        if not self.config.get("enable_phase_detection", True):
            return self.current_phase
            
        try:
            # Simple heuristic-based phase detection
            elbow_angle = metrics.get('front_elbow_angle', 0)
            wrist_velocity = metrics.get('right_wrist_velocity', 0)
            
            # Phase detection logic
            if frame_num < 10:
                phase = 'stance'
            elif wrist_velocity > 0.05 and elbow_angle > 120:
                phase = 'stride'
            elif wrist_velocity > 0.1 and elbow_angle < 110:
                phase = 'downswing'
            elif wrist_velocity > 0.15:
                phase = 'impact'
            elif wrist_velocity > 0.05:
                phase = 'follow_through'
            else:
                phase = 'recovery'
            
            if phase != self.current_phase:
                self.phase_transitions.append({
                    'frame': frame_num,
                    'from_phase': self.current_phase,
                    'to_phase': phase,
                    'timestamp': frame_num / 30.0  # Assuming 30 FPS
                })
                self.current_phase = phase
                
        except Exception as e:
            print(f"Error in phase detection: {e}")
            
        return self.current_phase
    
    def detect_contact_moment(self, metrics: Dict, frame_num: int):
        """Detect potential bat-ball contact using motion peaks"""
        if not self.config.get("enable_contact_detection", True):
            return
            
        try:
            wrist_velocity = metrics.get('right_wrist_velocity', 0)
            
            # Look for velocity spike indicating contact
            if wrist_velocity > 0.2 and self.contact_frame is None:
                self.contact_frame = frame_num
                print(f"Potential contact detected at frame {frame_num}")
                
        except Exception as e:
            print(f"Error in contact detection: {e}")
    
    def evaluate_frame_metrics(self, metrics: Dict) -> Dict[str, str]:
        """Evaluate frame metrics and provide real-time feedback"""
        feedback = {}
        thresholds = self.config["angle_thresholds"]
        
        # Elbow angle evaluation
        elbow_angle = metrics.get('front_elbow_angle', 0)
        if elbow_angle > 0:
            if thresholds["elbow_min"] <= elbow_angle <= thresholds["elbow_max"]:
                feedback['elbow'] = "✅ Good elbow elevation"
            else:
                feedback['elbow'] = "❌ Adjust elbow angle"
        
        # Spine lean evaluation
        spine_lean = metrics.get('spine_lean', 0)
        if spine_lean > 0:
            if spine_lean <= thresholds["spine_lean_max"]:
                feedback['spine'] = "✅ Good posture"
            else:
                feedback['spine'] = "❌ Excessive lean"
        
        # Head-knee alignment
        head_knee = metrics.get('head_knee_alignment', 0)
        if head_knee > 0:
            if head_knee <= thresholds["head_knee_max"]:
                feedback['head'] = "✅ Head over knee"
            else:
                feedback['head'] = "❌ Head not over front knee"
        
        # Balance evaluation
        balance = metrics.get('balance', 0)
        if balance < 5:
            feedback['balance'] = "✅ Good balance"
        elif balance > 10:
            feedback['balance'] = "❌ Poor balance"
        
        return feedback
    
    def draw_pose_overlay(self, frame: np.ndarray, landmarks: Dict, metrics: Dict, 
                         feedback: Dict, frame_num: int, current_phase: str) -> np.ndarray:
        """Draw comprehensive pose overlay with metrics and feedback"""
        overlay_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw pose skeleton
        if landmarks:
            # Define connections for skeleton
            connections = [
                ('nose', 'right_shoulder'), ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'), ('right_shoulder', 'right_hip'),
                ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
                ('left_shoulder', 'right_shoulder'), ('left_hip', 'right_hip')
            ]
            
            # Draw connections
            for start, end in connections:
                if start in landmarks and end in landmarks:
                    start_pos = (int(landmarks[start][0] * w), int(landmarks[start][1] * h))
                    end_pos = (int(landmarks[end][0] * w), int(landmarks[end][1] * h))
                    cv2.line(overlay_frame, start_pos, end_pos, (0, 255, 0), 2)
            
            # Draw key joints
            key_joints = ['nose', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_hip', 'right_knee']
            for joint in key_joints:
                if joint in landmarks:
                    pos = (int(landmarks[joint][0] * w), int(landmarks[joint][1] * h))
                    cv2.circle(overlay_frame, pos, 5, (255, 0, 0), -1)
        
        # Create overlay panel
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark background
        
        # Display metrics
        y_offset = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        
        # Frame info
        cv2.putText(panel, f"Frame: {frame_num} | Phase: {current_phase.title()}", 
                   (10, y_offset), font, font_scale, color, 1)
        y_offset += 25
        
        # Metrics display
        metric_displays = [
            ('Elbow Angle', metrics.get('front_elbow_angle', 0), '°'),
            ('Spine Lean', metrics.get('spine_lean', 0), '°'),
            ('Head-Knee', metrics.get('head_knee_alignment', 0), 'px'),
            ('Balance', metrics.get('balance', 0), '')
        ]
        
        for name, value, unit in metric_displays:
            if value > 0:
                cv2.putText(panel, f"{name}: {value:.1f}{unit}", 
                           (10, y_offset), font, font_scale, color, 1)
                y_offset += 20
        
        # Feedback display
        y_offset += 10
        for category, message in feedback.items():
            color = (0, 255, 0) if "✅" in message else (0, 0, 255)
            cv2.putText(panel, message, (10, y_offset), font, font_scale - 0.1, color, 1)
            y_offset += 20
        
        # Combine frame and panel
        combined_frame = np.vstack([overlay_frame, panel])
        
        return combined_frame
    
    def process_video(self, video_path: str, progress_callback=None) -> bool:
        """Process video with comprehensive analysis"""
        print(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"Error: Video file does not exist: {video_path}")
            return False
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Setup output video with better codec compatibility
        output_path = os.path.join(self.config["output_dir"], "annotated_video.mp4")
        
        # Try multiple codec options for better compatibility
        codecs_to_try = [
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
            ('H264', '.mp4'),
            ('MJPG', '.avi')
        ]
        
        out = None
        for codec, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = output_path.replace('.mp4', ext)
                out = cv2.VideoWriter(test_path, fourcc, fps, (width, height + 200))
                
                if out.isOpened():
                    output_path = test_path
                    print(f"Using codec: {codec}, output: {output_path}")
                    break
                else:
                    out.release()
                    out = None
            except:
                if out:
                    out.release()
                out = None
                continue
        
        if not out or not out.isOpened():
            print("Error: Could not initialize video writer with any codec")
            cap.release()
            return False
        
        frame_count = 0
        processing_start = time.time()
        
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            try:
                # MediaPipe processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                # Extract landmarks
                landmarks = self.extract_landmarks(results)
                
                # Calculate metrics
                metrics = self.calculate_biomechanical_metrics(landmarks, width, height)
                
                # Phase detection
                current_phase = self.detect_phase(metrics, frame_count)
                
                # Contact detection
                self.detect_contact_moment(metrics, frame_count)
                
                # Frame evaluation
                feedback = self.evaluate_frame_metrics(metrics)
                
                # Store metrics
                frame_data = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'phase': current_phase,
                    'metrics': metrics,
                    'feedback': feedback
                }
                self.frame_metrics.append(frame_data)
                
                # Draw overlay
                annotated_frame = self.draw_pose_overlay(
                    frame, landmarks, metrics, feedback, frame_count, current_phase
                )
                
                # Write frame - ensure correct dimensions
                if annotated_frame.shape[:2] == (height + 200, width):
                    out.write(annotated_frame)
                else:
                    # Resize if needed
                    resized_frame = cv2.resize(annotated_frame, (width, height + 200))
                    out.write(resized_frame)
                    
                frame_count += 1
                
                # Performance tracking
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_tracker.append(current_fps)
                
                # Progress callback
                if progress_callback and frame_count % 30 == 0:
                    progress = min(90, (frame_count / total_frames) * 100)
                    progress_callback(progress)
                
                # Console output
                if frame_count % 60 == 0:
                    avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
                    print(f"Processed {frame_count}/{total_frames} frames, avg FPS: {avg_fps:.1f}")
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                continue
        
        # Cleanup
        cap.release()
        out.release()
        
        processing_time = time.time() - processing_start
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"Processing completed in {processing_time:.2f}s")
        print(f"Average processing FPS: {avg_fps:.2f}")
        print(f"Output saved to: {output_path}")
        
        # Verify output file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"✅ Video successfully created: {os.path.getsize(output_path)/1024/1024:.1f} MB")
            
            # Update config with actual output path
            self.config["output_video_path"] = output_path
            
            # Generate smoothness chart if enabled
            if self.config.get("enable_smoothness_analysis", True):
                self.generate_smoothness_chart()
            
            return True
        else:
            print("❌ Video creation failed - file empty or missing")
            return False
    
    def generate_smoothness_chart(self):
        """Generate smoothness analysis chart"""
        try:
            if len(self.frame_metrics) < 10:
                return
            
            # Extract time series data
            timestamps = [fm['timestamp'] for fm in self.frame_metrics]
            elbow_angles = [fm['metrics'].get('front_elbow_angle', 0) for fm in self.frame_metrics]
            spine_leans = [fm['metrics'].get('spine_lean', 0) for fm in self.frame_metrics]
            
            # Filter out zero values
            valid_indices = [i for i, (e, s) in enumerate(zip(elbow_angles, spine_leans)) if e > 0 and s > 0]
            
            if len(valid_indices) < 5:
                return
            
            timestamps = [timestamps[i] for i in valid_indices]
            elbow_angles = [elbow_angles[i] for i in valid_indices]
            spine_leans = [spine_leans[i] for i in valid_indices]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Elbow angle plot
            ax1.plot(timestamps, elbow_angles, 'b-', linewidth=2, label='Elbow Angle')
            ax1.axhline(y=115, color='g', linestyle='--', alpha=0.7, label='Ideal (115°)')
            ax1.fill_between(timestamps, 90, 140, alpha=0.2, color='green', label='Target Range')
            ax1.set_ylabel('Elbow Angle (degrees)')
            ax1.set_title('Elbow Angle Consistency Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Spine lean plot
            ax2.plot(timestamps, spine_leans, 'r-', linewidth=2, label='Spine Lean')
            ax2.axhline(y=5, color='g', linestyle='--', alpha=0.7, label='Ideal (5°)')
            ax2.fill_between(timestamps, 0, 15, alpha=0.2, color='green', label='Target Range')
            ax2.set_ylabel('Spine Lean (degrees)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_title('Spine Lean Consistency Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.config["output_dir"], "smoothness_analysis.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Smoothness chart saved to: {chart_path}")
            
        except Exception as e:
            print(f"Error generating smoothness chart: {e}")
            
    
    def calculate_final_scores(self) -> Dict:
        """Calculate final comprehensive scores"""
        if not self.frame_metrics:
            return self.get_default_scores()
        
        # Extract metrics for analysis
        elbow_angles = [fm['metrics'].get('front_elbow_angle', 0) for fm in self.frame_metrics if fm['metrics'].get('front_elbow_angle', 0) > 0]
        spine_leans = [fm['metrics'].get('spine_lean', 0) for fm in self.frame_metrics if fm['metrics'].get('spine_lean', 0) > 0]
        head_knee_distances = [fm['metrics'].get('head_knee_alignment', 0) for fm in self.frame_metrics if fm['metrics'].get('head_knee_alignment', 0) > 0]
        balance_scores = [fm['metrics'].get('balance', 0) for fm in self.frame_metrics if fm['metrics'].get('balance', 0) >= 0]
        
        scores = {}
        feedback = {}
        
        # 1. Swing Control (based on elbow angle consistency)
        if elbow_angles:
            avg_elbow = np.mean(elbow_angles)
            elbow_consistency = 1.0 - (np.std(elbow_angles) / 50.0)  # Normalize std
            target_deviation = abs(avg_elbow - 115) / 25.0  # Ideal elbow angle is 115°
            swing_score = max(1, min(10, (1 - target_deviation) * 10 * elbow_consistency))
            scores['swing_control'] = round(swing_score, 1)
            
            if avg_elbow >= 90 and avg_elbow <= 140:
                feedback['swing_control'] = ["Good elbow angle range maintained", "Focus on consistency through the shot"]
            else:
                feedback['swing_control'] = ["Work on elbow positioning", f"Current average: {avg_elbow:.1f}°, target: 115°"]
        else:
            scores['swing_control'] = 5.0
            feedback['swing_control'] = ["Elbow angle analysis unclear - ensure side view"]
        
        # 2. Balance (based on hip alignment and stability)
        if balance_scores:
            avg_balance = np.mean(balance_scores)
            balance_score = max(1, min(10, 10 - avg_balance))
            scores['balance'] = round(balance_score, 1)
            
            if avg_balance < 3:
                feedback['balance'] = ["Excellent balance maintained", "Good stability throughout shot"]
            elif avg_balance < 7:
                feedback['balance'] = ["Decent balance", "Focus on maintaining hip alignment"]
            else:
                feedback['balance'] = ["Work on balance and stability", "Focus on keeping hips level"]
        else:
            scores['balance'] = 6.0
            feedback['balance'] = ["Balance analysis needs clearer hip visibility"]
        
        # 3. Head Position (based on head-knee alignment)
        if head_knee_distances:
            avg_head_knee = np.mean(head_knee_distances)
            head_score = max(1, min(10, 10 - (avg_head_knee / 10)))
            scores['head_position'] = round(head_score, 1)
            
            if avg_head_knee < 20:
                feedback['head_position'] = ["Good head positioning", "Head well positioned over front leg"]
            else:
                feedback['head_position'] = ["Improve head position", "Try to keep head over front knee"]
        else:
            scores['head_position'] = 6.0
            feedback['head_position'] = ["Head position analysis needs better angle"]
        
        # 4. Footwork (based on phase transitions and foot positioning)
        phase_count = len(self.phase_transitions)
        if phase_count >= 3:  # Good phase progression
            footwork_score = min(10, 6 + phase_count)
            scores['footwork'] = round(footwork_score, 1)
            feedback['footwork'] = ["Good shot progression through phases",
                                     "Smooth transitions between stance and execution"]
        else:
            scores['footwork'] = 5.0
            feedback['footwork'] = ["Work on shot timing and foot movement",
                                  "Focus on smooth phase transitions"]
        
        # 5. Follow-through (based on final phase data)
        follow_through_frames = [fm for fm in self.frame_metrics if fm['phase'] in ['follow_through', 'recovery']]
        if len(follow_through_frames) >= 5:
            follow_through_score = 8.0
            scores['follow_through'] = follow_through_score
            feedback['follow_through'] = ["Good follow-through completion",
                                        "Shot completed with proper extension"]
        elif len(follow_through_frames) >= 2:
            scores['follow_through'] = 6.0
            feedback['follow_through'] = ["Adequate follow-through",
                                        "Could extend more through the shot"]
        else:
            scores['follow_through'] = 4.0
            feedback['follow_through'] = ["Incomplete follow-through detected",
                                        "Focus on completing the shot fully"]
        
        return {'scores': scores, 'feedback': feedback}
    
    
    def get_default_scores(self) -> Dict:
        """Return default scores when analysis fails"""
        return {
            'scores': {
                'swing_control': 5.0,
                'balance': 5.0,
                'head_position': 5.0,
                'footwork': 5.0,
                'follow_through': 5.0
            },
            'feedback': {
                'swing_control': ["Analysis incomplete - ensure clear side view"],
                'balance': ["Balance assessment needs better visibility"],
                'head_position': ["Head position unclear in video"],
                'footwork': ["Footwork analysis needs full body visibility"],
                'follow_through': ["Follow-through not clearly visible"]
            }
        }
    
    def determine_skill_grade(self, scores: Dict) -> str:
        """Determine skill grade based on scores"""
        avg_score = np.mean(list(scores.values()))
        
        if avg_score >= 8.0:
            return "Advanced"
        elif avg_score >= 6.0:
            return "Intermediate"
        else:
            return "Beginner"
    
    def compare_with_reference(self, scores: Dict) -> Dict:
        """Compare with reference/ideal cricket technique"""
        reference = self.config.get("reference_comparison", {})
        deviations = {}
        
        if self.frame_metrics:
            # Calculate average metrics
            elbow_angles = [fm['metrics'].get('front_elbow_angle', 0) for fm in self.frame_metrics if fm['metrics'].get('front_elbow_angle', 0) > 0]
            spine_leans = [fm['metrics'].get('spine_lean', 0) for fm in self.frame_metrics if fm['metrics'].get('spine_lean', 0) > 0]
            
            if elbow_angles:
                avg_elbow = np.mean(elbow_angles)
                ideal_elbow = reference.get('elbow_ideal', 115)
                deviations['elbow_deviation'] = round(avg_elbow - ideal_elbow, 1)
            
            if spine_leans:
                avg_spine = np.mean(spine_leans)
                ideal_spine = reference.get('spine_lean_ideal', 5)
                deviations['spine_deviation'] = round(avg_spine - ideal_spine, 1)
        
        return deviations
    
    def generate_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        evaluation_data = self.calculate_final_scores()
        scores = evaluation_data.get('scores', {})
        feedback = evaluation_data.get('feedback', {})
        
        # Additional analysis
        skill_grade = self.determine_skill_grade(scores)
        reference_comparison = self.compare_with_reference(scores)
        
        # Performance metrics
        avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0
        
        report = {
            "summary": {
                "total_frames_analyzed": len(self.frame_metrics),
                "video_duration": len(self.frame_metrics) / 30.0 if self.frame_metrics else 0,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "skill_grade": skill_grade,
                "contact_frame_detected": self.contact_frame,
                "phase_transitions": len(self.phase_transitions),
                "average_processing_fps": round(avg_fps, 2)
            },
            "scores": scores,
            "feedback": feedback,
            "phase_analysis": {
                "phases_detected": [pt['to_phase'] for pt in self.phase_transitions],
                "phase_transitions": self.phase_transitions
            },
            "reference_comparison": reference_comparison,
            "performance_metrics": {
                "target_fps_achieved": avg_fps >= self.config.get("target_fps", 10),
                "real_time_capable": avg_fps >= 15
            }
        }
        
        return report
    
    def save_evaluation(self, evaluation: Dict):
        """Save evaluation to JSON file"""
        output_path = os.path.join(self.config["output_dir"], "evaluation.json")
        
        with open(output_path, 'w') as f:
            json.dump(evaluation, f, indent=2, default=str)
        
        # Also save a text summary
        text_path = os.path.join(self.config["output_dir"], "evaluation.txt")
        self.save_text_summary(evaluation, text_path)
        
        print(f"Evaluation saved to: {output_path}")
        print(f"Text summary saved to: {text_path}")
    
    def save_text_summary(self, evaluation: Dict, filepath: str):
        """Save human-readable text summary"""
        with open(filepath, 'w') as f:
            f.write("=== AthleteRise Cricket Analysis Report ===\n\n")
            
            # Summary
            summary = evaluation.get('summary', {})
            f.write(f"Analysis Date: {summary.get('analysis_timestamp', 'N/A')}\n")
            f.write(f"Skill Grade: {summary.get('skill_grade', 'N/A')}\n")
            f.write(f"Processing FPS: {summary.get('average_processing_fps', 0):.1f}\n\n")
            
            # Scores
            scores = evaluation.get('scores', {})
            f.write("=== PERFORMANCE SCORES ===\n")
            for category, score in scores.items():
                f.write(f"{category.replace('_', ' ').title()}: {score}/10\n")
            f.write(f"\nOverall Average: {np.mean(list(scores.values())):.1f}/10\n\n")
            
            # Feedback
            feedback = evaluation.get('feedback', {})
            f.write("=== DETAILED FEEDBACK ===\n")
            for category, comments in feedback.items():
                f.write(f"\n{category.replace('_', ' ').title()}:\n")
                for comment in comments:
                    f.write(f"  • {comment}\n")
            
            # Phase Analysis
            phase_data = evaluation.get('phase_analysis', {})
            if phase_data.get('phases_detected'):
                f.write(f"\n=== PHASE ANALYSIS ===\n")
                f.write(f"Phases Detected: {', '.join(phase_data['phases_detected'])}\n")
            
            # Reference Comparison
            ref_comp = evaluation.get('reference_comparison', {})
            if ref_comp:
                f.write(f"\n=== TECHNIQUE COMPARISON ===\n")
                for metric, deviation in ref_comp.items():
                    f.write(f"{metric}: {deviation:+.1f}° from ideal\n")


def create_default_config():
    """Create a default configuration file"""
    config = {
        "video_url": "https://youtube.com/shorts/vSX3IRxGnNY",
        "input_path": "input/cricket_video.mp4",
        "output_dir": "output",
        "pose_confidence": 0.7,
        "pose_detection_confidence": 0.6,
        "angle_thresholds": {
            "elbow_min": 90,
            "elbow_max": 140,
            "spine_lean_max": 15,
            "head_knee_max": 30,
            "foot_angle_optimal": 45
        },
        "fps_target": 15,
        "model_complexity": 1,
        "enable_phase_detection": True,
        "enable_contact_detection": True,
        "enable_smoothness_analysis": True,
        "target_fps": 10.0,
        "reference_comparison": {
            "elbow_ideal": 115,
            "spine_lean_ideal": 5,
            "head_knee_ideal": 10
        }
    }
    
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Default configuration created: config.json")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='AthleteRise Cricket Analysis')
    parser.add_argument('--video-url', type=str, 
                       default='https://youtube.com/shorts/vSX3IRxGnNY',
                       help='YouTube video URL to analyze')
    parser.add_argument('--input-path', type=str,
                       help='Local video file path')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        create_default_config()
        return
    
    try:
        # Initialize analyzer
        analyzer = CricketPoseAnalyzer(args.config)
        
        # Update config with command line args
        if args.output_dir:
            analyzer.config["output_dir"] = args.output_dir
        
        video_path = None
        
        # Determine video source
        if args.input_path and os.path.exists(args.input_path):
            video_path = args.input_path
            print(f"Using local video: {video_path}")
        elif args.video_url:
            print(f"Downloading from URL: {args.video_url}")
            video_path = analyzer.download_video(args.video_url)
            if not video_path:
                print("Failed to download video")
                return
        else:
            print("Error: No valid video source provided")
            print("Use --input-path for local video or --video-url for YouTube")
            return
        
        # Process video
        print("Starting video analysis...")
        success = analyzer.process_video(video_path)
        
        if success:
            # Generate evaluation
            print("Generating evaluation report...")
            evaluation = analyzer.generate_evaluation_report()
            analyzer.save_evaluation(evaluation)
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE!")
            print("="*60)
            
            # Print summary
            summary = evaluation.get('summary', {})
            print(f"📊 Skill Grade: {summary.get('skill_grade', 'N/A')}")
            print(f"⚡ Processing FPS: {summary.get('average_processing_fps', 0):.1f}")
            print(f"🎯 Frames Analyzed: {summary.get('total_frames_analyzed', 0)}")
            
            scores = evaluation.get('scores', {})
            overall_score = np.mean(list(scores.values()))
            print(f"🏏 Overall Score: {overall_score:.1f}/10")
            
            # Print individual scores
            print("\n📈 Category Scores:")
            for category, score in scores.items():
                status = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
                print(f"  {status} {category.replace('_', ' ').title()}: {score}/10")
            
            # Print key feedback
            feedback = evaluation.get('feedback', {})
            if feedback:
                print("\n💡 Key Recommendations:")
                for category, comments in feedback.items():
                    if comments and len(comments) > 0:
                        print(f"  • {comments[0]}")
            
            print(f"\n📁 Output files saved to: {analyzer.config['output_dir']}/")
            print("   - annotated_video.mp4 (or .avi)")
            print("   - evaluation.json")
            print("   - evaluation.txt")
            print("   - smoothness_analysis.png")
            
        else:
            print("❌ Analysis failed - check video file and dependencies")
            
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()