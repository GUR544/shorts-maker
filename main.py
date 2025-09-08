import streamlit as st
import os
import uuid
import yt_dlp
import cv2
import numpy as np
import shutil
import zipfile
import io
import subprocess

# --- 1. APP CONFIGURATION & STYLING ---

st.set_page_config(
    page_title="Shorts Maker AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def local_css():
    """Injects custom CSS for a premium UI."""
    css = """
    <style>
        /* General Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
        body, .stApp {
            font-family: 'Inter', sans-serif;
            background: #121212;
            color: #ffffff;
        }
        #MainMenu, .stDeployButton, footer { visibility: hidden; }
        header { visibility: hidden; height: 0; margin: 0; }

        /* Main Container Styling */
        .main-container {
            max-width: 800px;
            margin: 2rem auto;
            padding: 2.5rem;
            background: rgba(30, 30, 30, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(10px);
        }

        /* Title and Text Styling */
        h1 {
            font-size: 3rem;
            font-weight: 900;
            text-align: center;
            background: linear-gradient(90deg, #8A2BE2, #4B0082, #FF6B6B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .st-emotion-cache-10trblm, .st-emotion-cache-1629p8f h2 {
            text-align: center;
            color: #b3b3b3;
            font-weight: 400;
        }

        /* Input and Button Styling */
        .stTextInput > div > div > input {
            background-color: rgba(0, 0, 0, 0.2);
            color: white;
            border-radius: 12px;
            padding: 1rem;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: linear-gradient(90deg, #8A2BE2, #4B0082);
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            transition: transform 0.2s;
        }
        div[data-testid="stButton"] > button:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(138, 43, 226, 0.4);
        }

        /* Custom Cards for Ratio/FPS Selection */
        .option-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .option-card:hover { border-color: #8A2BE2; }
        .option-card.selected-card {
            border-color: #1DB954;
            background: rgba(29, 185, 84, 0.1);
        }
        .option-card h3 { margin-top: 0; }
        .option-card p { font-size: 0.9rem; color: #b3b3b3; margin-bottom: 0; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 2. BACKEND HELPER FUNCTIONS ---

TEMP_DIR = "temp_shorts_maker"

def setup_directories():
    """Create or clean up temporary directories."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def get_video_info(url):
    """Fetches video title and thumbnail without downloading."""
    ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {"title": info.get("title", "Unknown Title"), "thumbnail": info.get("thumbnail", None)}
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def download_video(_url):
    """Downloads the best quality video from a YouTube URL."""
    video_id = str(uuid.uuid4())
    output_path = os.path.join(TEMP_DIR, f"{video_id}.mp4")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([_url])
    return output_path

def detect_highlights(video_path, num_clips=5):
    """Detects highlights using OpenCV based on scene changes."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scene_changes = []
    ret, prev_frame = cap.read()
    if not ret: return []
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Check every half-second to speed up processing
        if frame_count % int(fps / 2) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_frame_gray, gray)
            if np.mean(diff) > 15: # Heuristic for scene change
                scene_changes.append(frame_count / fps)
            prev_frame_gray = gray
        frame_count += 1
        
    cap.release()

    if not scene_changes: # Fallback: split video evenly
        duration = total_frames / fps
        scene_changes = [i * duration / num_clips for i in range(num_clips)]
    
    # Create clips around detected scene changes
    clips = []
    duration = total_frames / fps
    for t in sorted(list(set(scene_changes)))[:num_clips]:
        start_time = max(0, t - 5)
        end_time = min(duration, start_time + 45) # Create ~45s clips
        if end_time - start_time > 10: # Ensure clip is reasonably long
            clips.append((start_time, end_time))
            
    return clips

def create_short_with_ffmpeg(video_path, start, end, aspect_ratio, fps, clip_index):
    """
    Generates a short clip using FFMPEG command-line tool.
    This function replaces all moviepy logic.
    """
    output_path = os.path.join(TEMP_DIR, f"short_{clip_index}.mp4")
    duration = end - start

    # Base FFMPEG command
    command = [
        'ffmpeg',
        '-ss', str(start),        # Seek to start time
        '-t', str(duration),      # Set duration of the clip
        '-i', video_path,         # Input file
        '-loglevel', 'error',     # Suppress verbose output
        '-y',                     # Overwrite output file if it exists
    ]
    
    # Define video filters based on aspect ratio
    if aspect_ratio == "9:16":
        # Complex filter for vertical video with blurred background
        filter_complex = (
            "[0:v]split[main][bg];"  # Split video stream into two
            "[bg]crop=ih*9/16:ih,scale=1080:1920,boxblur=10[bg_blurred];" # Create blurred background
            "[main]scale=1080:-1[fg];" # Scale main video to fit width
            "[bg_blurred][fg]overlay=(W-w)/2:(H-h)/2,fps={}".format(fps) # Overlay foreground on background
        )
        command.extend(['-filter_complex', filter_complex, '-c:a', 'copy'])
    else:
        # Standard crop and scale for other ratios
        ratios = {"16:9": "1920:1080", "1:1": "1080:1080", "4:3": "1440:1080", "21:9": "1920:822"}
        target_res = ratios.get(aspect_ratio, "1920:1080")
        vf_command = f"scale={target_res}:force_original_aspect_ratio=decrease,pad={target_res}:-1:-1:color=black,fps={fps}"
        command.extend(['-vf', vf_command, '-c:a', 'copy'])

    command.append(output_path)

    try:
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"FFMPEG Error while creating clip {clip_index}: {e}")
        return None

# --- 3. STREAMLIT UI FLOW ---

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"
    st.session_state.url = ""
    st.session_state.aspect_ratio = None
    st.session_state.fps = None
    st.session_state.generated_clips = []
    st.session_state.video_info = None

local_css()

# --- LANDING WINDOW ---
if st.session_state.page == "landing":
    setup_directories()
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("Shorts Maker AI")
    st.header("Turn any YouTube video into viral short clips instantly.")
    url = st.text_input("Paste your YouTube video link here...", key="youtube_url", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Analyze Video"):
        if url:
            with st.spinner("Fetching video details..."):
                video_info = get_video_info(url)
                if video_info:
                    st.session_state.url, st.session_state.video_info = url, video_info
                    st.session_state.page = "options"
                    st.rerun()
                else:
                    st.error("Invalid YouTube URL or video not accessible.")
        else:
            st.warning("Please enter a URL.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- OPTIONS WINDOW ---
elif st.session_state.page == "options":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("Customize Your Shorts")
    st.header(f"Video: {st.session_state.video_info['title']}")
    if st.session_state.video_info['thumbnail']:
        st.image(st.session_state.video_info['thumbnail'], use_column_width=True)

    st.subheader("1. Choose Aspect Ratio")
    ratios = { "9:16": ("Vertical", "YouTube Shorts, TikTok"), "16:9": ("Landscape", "Standard YouTube"), "1:1": ("Square", "Social Feeds"), "4:3": ("Retro", "Classic TV") }
    cols = st.columns(len(ratios))
    for i, (ratio, (name, desc)) in enumerate(ratios.items()):
        with cols[i]:
            is_selected = st.session_state.aspect_ratio == ratio
            st.markdown(f'<div class="option-card {"selected-card" if is_selected else ""}" onclick="document.getElementById(\'btn_ratio_{ratio}\').click()"><h3>{ratio}</h3><p>{name} / {desc}</p></div>', unsafe_allow_html=True)
            if st.button(f"Select_{ratio}", key=f"btn_ratio_{ratio}", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.aspect_ratio = ratio
                st.rerun()

    st.subheader("2. Select Frames Per Second (FPS)")
    fps_options = { 40: "Smaller Files", 60: "Ultra Smooth", 90: "Cinematic" }
    cols = st.columns(len(fps_options))
    for i, (fps, desc) in enumerate(fps_options.items()):
        with cols[i]:
            is_selected = st.session_state.fps == fps
            st.markdown(f'<div class="option-card {"selected-card" if is_selected else ""}" onclick="document.getElementById(\'btn_fps_{fps}\').click()"><h3>{fps} FPS</h3><p>{desc}</p></div>', unsafe_allow_html=True)
            if st.button(f"Select_{fps} FPS", key=f"btn_fps_{fps}", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.fps = fps
                st.rerun()

    st.write("---")
    is_ready = st.session_state.aspect_ratio and st.session_state.fps
    if st.button("Generate Clips", disabled=not is_ready):
        st.session_state.page = "processing"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# --- PROCESSING & PREVIEW WINDOW ---
elif st.session_state.page in ["processing", "preview"]:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("Your Clips Are Ready!")

    if not st.session_state.generated_clips:
        with st.spinner("Downloading, analyzing, and rendering clips with FFMPEG... This may take a few minutes."):
            video_path = download_video(st.session_state.url)
            highlights = detect_highlights(video_path)
            
            clips = []
            progress_bar = st.progress(0, text="Rendering clips...")
            for i, (start, end) in enumerate(highlights):
                clip_path = create_short_with_ffmpeg(
                    video_path, start, end, 
                    st.session_state.aspect_ratio, 
                    st.session_state.fps, i
                )
                if clip_path: clips.append(clip_path)
                progress_bar.progress((i + 1) / len(highlights), f"Rendered clip {i+1}/{len(highlights)}")
            
            st.session_state.generated_clips = clips
            st.session_state.page = "preview"
            st.rerun()

    if st.session_state.page == "preview":
        st.header("Preview and select clips for download.")
        selections = {clip_path: st.checkbox(f"Select Clip #{i+1}", key=f"cb_{i}") for i, clip_path in enumerate(st.session_state.generated_clips)}
        
        for clip_path in st.session_state.generated_clips:
            st.video(clip_path)
        
        selected_clips = [path for path, selected in selections.items() if selected]
        
        if selected_clips:
            if len(selected_clips) == 1:
                with open(selected_clips[0], "rb") as f:
                    st.download_button("Download Selected Clip", f, file_name=os.path.basename(selected_clips[0]), mime="video/mp4", use_container_width=True)
            else:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for path in selected_clips: zf.write(path, os.path.basename(path))
                st.download_button(f"Download {len(selected_clips)} Clips (.zip)", zip_buffer, file_name="shorts_clips.zip", mime="application/zip", use_container_width=True)

    if st.button("Start Over"):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
