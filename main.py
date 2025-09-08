import streamlit as st
import os
import uuid
import yt_dlp
import moviepy.editor as mp
from moviepy.video.fx.all import crop, resize
import cv2
import numpy as np
import shutil
import zipfile
import io

# --- 1. APP CONFIGURATION & STYLING ---

# Configure the Streamlit page
st.set_page_config(
    page_title="Shorts Maker AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to inject custom CSS for a premium look
def local_css():
    css = """
    <style>
        /* General Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

        body, .stApp {
            font-family: 'Inter', sans-serif;
            background: #121212;
            color: #ffffff;
        }

        /* Hide Streamlit Header/Footer */
        #MainMenu, .stDeployButton, footer {
            visibility: hidden;
        }
        header {
            visibility: hidden;
            height: 0;
            margin: 0;
        }

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
            -webkit-backdrop-filter: blur(10px);
        }

        /* Title and Text Styling */
        h1 {
            font-size: 3rem;
            font-weight: 900;
            text-align: center;
            background: linear-gradient(90deg, #8A2BE2, #4B0082, #FF6B6B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
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
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1rem;
            transition: all 0.2s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: #8A2BE2;
            box-shadow: 0 0 15px rgba(138, 43, 226, 0.5);
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
            transition: all 0.3s ease;
        }
        div[data-testid="stButton"] > button:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 5px 20px rgba(138, 43, 226, 0.4);
        }
        div[data-testid="stButton"] > button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
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
        .option-card:hover {
            border-color: #8A2BE2;
        }
        .option-card h3 {
            margin-top: 0;
            font-weight: 600;
        }
        .option-card p {
            font-size: 0.9rem;
            color: #b3b3b3;
            margin-bottom: 0;
        }
        .selected-card {
            border-color: #1DB954;
            background: rgba(29, 185, 84, 0.1);
        }

        /* Video Preview Grid */
        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1.5rem;
        }
        .video-container {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 2. BACKEND HELPER FUNCTIONS ---

# Setup temporary directories for video processing
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
    """
    Detects highlights based on scene changes.
    A simple heuristic: significant changes in frame content indicate a scene cut.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scene_changes = []
    prev_frame = None
    
    ret, frame = cap.read()
    if ret:
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check every 'fps' frames (approx 1 second) to speed up processing
        if frame_count % int(fps) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_frame, gray)
            non_zero_count = np.count_nonzero(diff)
            
            # Threshold for scene change detection (heuristic)
            if non_zero_count > (gray.shape[0] * gray.shape[1] * 0.25):
                scene_changes.append(frame_count / fps)
            
            prev_frame = gray
            
        frame_count += 1
        
    cap.release()

    if not scene_changes: # If no major scenes, just split video evenly
        duration = total_frames / fps
        for i in range(num_clips):
            start = (i / num_clips) * duration
            scene_changes.append(start)
    
    # Create clips around scene changes (e.g., 5s before to 25s after)
    clips = []
    for t in sorted(list(set(scene_changes)))[:num_clips]:
        start_time = max(0, t - 5)
        end_time = start_time + 30 # Create 30-second clips
        clips.append((start_time, end_time))
        
    return clips

def create_short(video_path, start, end, aspect_ratio_str, fps, clip_index):
    """Generates a single short clip with specified transformations."""
    video = mp.VideoFileClip(video_path).subclip(start, end)
    
    w, h = video.size
    target_ratio = {"9:16": 9/16, "16:9": 16/9, "1:1": 1.0, "4:3": 4/3, "21:9": 21/9}[aspect_ratio_str]
    target_w, target_h = (1080, 1920) if aspect_ratio_str == "9:16" else (1920, 1080)

    # Handle 9:16 vertical format with blurred background
    if aspect_ratio_str == "9:16" and w / h > target_ratio:
        scaled_clip = video.resize(width=target_w)
        
        # Create blurred background
        bg_clip = crop(video, width=int(h * target_ratio), x_center=w/2)
        bg_clip = bg_clip.resize(height=target_h).fx(mp.vfx.blur, radius=20)
        
        final_clip = mp.CompositeVideoClip(
            [bg_clip, scaled_clip.set_position("center")],
            size=(target_w, target_h)
        )
    else: # Standard crop and resize for other formats
        final_clip = crop(video, x_center=w/2, y_center=h/2, 
                          width=min(w, int(h * target_ratio)), 
                          height=min(h, int(w / target_ratio)))
        final_clip = final_clip.resize((target_w, target_h))

    final_clip = final_clip.set_fps(fps).set_duration(final_clip.duration)
    
    output_path = os.path.join(TEMP_DIR, f"short_{clip_index}.mp4")
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", threads=4)
    video.close()
    return output_path


# --- 3. STREAMLIT UI FLOW ---

# Initialize session state variables
if "page" not in st.session_state:
    st.session_state.page = "landing"
    st.session_state.url = ""
    st.session_state.aspect_ratio = None
    st.session_state.fps = None
    st.session_state.generated_clips = []
    st.session_state.video_info = None

# Apply custom CSS
local_css()

# --- LANDING WINDOW ---
if st.session_state.page == "landing":
    setup_directories() # Clean up on first load
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("Shorts Maker AI")
    st.header("Turn any YouTube video into viral short clips instantly.")
    
    url = st.text_input("Paste your YouTube video link here...", key="youtube_url", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Analyze Video"):
        if url:
            with st.spinner("Fetching video details..."):
                video_info = get_video_info(url)
                if video_info:
                    st.session_state.url = url
                    st.session_state.video_info = video_info
                    st.session_state.page = "options"
                    st.rerun()
                else:
                    st.error("Invalid YouTube URL or video is not accessible. Please check the link.")
        else:
            st.warning("Please enter a YouTube URL.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- OPTIONS WINDOW ---
elif st.session_state.page == "options":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("Customize Your Shorts")
    st.header(f"Video: {st.session_state.video_info['title']}")
    
    if st.session_state.video_info['thumbnail']:
        st.image(st.session_state.video_info['thumbnail'], use_column_width=True)

    st.subheader("1. Choose Aspect Ratio")
    ratios = {
        "9:16": ("Vertical", "For YouTube Shorts, TikTok, and Reels. Fills mobile screens."),
        "16:9": ("Landscape", "Standard for desktop YouTube videos and films."),
        "1:1": ("Square", "Great for social media feed posts (Instagram, Facebook)."),
        "4:3": ("Retro", "Classic TV format for a vintage look."),
    }
    
    cols = st.columns(len(ratios))
    for i, (ratio, (name, desc)) in enumerate(ratios.items()):
        with cols[i]:
            selected_class = "selected-card" if st.session_state.aspect_ratio == ratio else ""
            st.markdown(f"""
            <div class="option-card {selected_class}" id="ratio-{ratio}">
                <h3>{ratio}</h3>
                <p>{name}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select {ratio}", key=f"btn_ratio_{ratio}", use_container_width=True):
                st.session_state.aspect_ratio = ratio
                st.rerun()

    st.subheader("2. Select Frames Per Second (FPS)")
    fps_options = {
        40: "Good balance for size and older devices.",
        60: "The standard for smooth, high-quality video.",
        90: "Excellent for slow-motion or a premium cinematic feel."
    }
    
    cols = st.columns(len(fps_options))
    for i, (fps, desc) in enumerate(fps_options.items()):
        with cols[i]:
            selected_class = "selected-card" if st.session_state.fps == fps else ""
            st.markdown(f"""
            <div class="option-card {selected_class}">
                <h3>{fps} FPS</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select {fps} FPS", key=f"btn_fps_{fps}", use_container_width=True):
                st.session_state.fps = fps
                st.rerun()

    st.write("---")

    # Proceed button enabled only when both options are selected
    is_ready = st.session_state.aspect_ratio and st.session_state.fps
    if st.button("Generate Clips", disabled=not is_ready, use_container_width=True):
        st.session_state.page = "processing"
        st.rerun()
    if not is_ready:
        st.info("Please select both an aspect ratio and FPS to continue.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- PROCESSING & PREVIEW WINDOW ---
elif st.session_state.page in ["processing", "preview"]:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("Your Clips Are Ready!")

    if not st.session_state.generated_clips:
        with st.spinner("This might take a few minutes... Downloading video, finding highlights, and rendering clips..."):
            try:
                video_path = download_video(st.session_state.url)
                highlights = detect_highlights(video_path)
                
                clips = []
                progress_bar = st.progress(0, text="Rendering clips...")
                for i, (start, end) in enumerate(highlights):
                    clip_path = create_short(
                        video_path, start, end, 
                        st.session_state.aspect_ratio, 
                        st.session_state.fps, i
                    )
                    clips.append(clip_path)
                    progress_bar.progress((i + 1) / len(highlights), text=f"Rendered clip {i+1}/{len(highlights)}")
                
                st.session_state.generated_clips = clips
                st.session_state.page = "preview"
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.session_state.page = "options" # Go back
                if st.button("Try Again"):
                    st.rerun()

    if st.session_state.page == "preview":
        st.header("Preview and select the clips you want to download.")
        
        selections = []
        st.markdown('<div class="video-grid">', unsafe_allow_html=True)
        for i, clip_path in enumerate(st.session_state.generated_clips):
            with st.container():
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                st.video(clip_path)
                st.markdown('</div>', unsafe_allow_html=True)
                if st.checkbox("Select for download", key=f"clip_select_{i}"):
                    selections.append(clip_path)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.write("---")

        if selections:
            if len(selections) == 1:
                with open(selections[0], "rb") as file:
                    st.download_button(
                        label=f"Download Selected Clip",
                        data=file,
                        file_name=os.path.basename(selections[0]),
                        mime="video/mp4",
                        use_container_width=True
                    )
            else:
                # Zip multiple files for download
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for clip_path in selections:
                        zip_file.write(clip_path, os.path.basename(clip_path))
                
                st.download_button(
                    label=f"Download {len(selections)} Selected Clips (.zip)",
                    data=zip_buffer,
                    file_name="shorts_maker_clips.zip",
                    mime="application/zip",
                    use_container_width=True
                )
    
    if st.button("Start Over"):
        # Clean up and reset
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
