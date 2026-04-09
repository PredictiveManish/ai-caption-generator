import os
import tempfile
import streamlit as st
from datetime import timedelta
import whisper
import srt
from moviepy import VideoFileClip, CompositeVideoClip, TextClip
import numpy as np
from pydub import AudioSegment
import base64
import time

# Page setup
st.set_page_config(
    page_title="Video Caption Maker", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Eye-soothing, sophisticated color palette
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #334155;
        --bg-hover: #475569;
        --accent-primary: #06b6d4;
        --accent-secondary: #8b5cf6;
        --accent-gradient: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --border: rgba(148, 163, 184, 0.1);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: var(--text-primary);
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }
    
    /* Glassmorphism cards */
    .card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }
    
    /* Step indicators with glass effect */
    .step-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        margin: 10px 0;
        background: rgba(51, 65, 85, 0.5);
        border-radius: 12px;
        border-left: 3px solid var(--accent-primary);
        transition: all 0.3s ease;
    }
    
    .step-item:hover {
        background: rgba(71, 85, 105, 0.6);
        border-left-width: 4px;
    }
    
    .step-number {
        background: var(--accent-gradient);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 14px;
        margin-right: 14px;
        box-shadow: 0 4px 6px -1px rgba(6, 182, 212, 0.3);
    }
    
    .step-text {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Success message with glow */
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
    }
    
    /* Info box with gradient border */
    .info-box {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid transparent;
        background-clip: padding-box;
        position: relative;
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        z-index: -1;
        margin: -1px;
        border-radius: inherit;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    }
    
    .info-box h3 {
        color: var(--accent-primary);
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .info-box h4 {
        color: var(--text-secondary);
        margin-top: 1.5rem;
        font-weight: 500;
    }
    
    .info-box ul, .info-box ol {
        color: var(--text-secondary);
        line-height: 1.8;
    }
    
    .info-box li {
        margin: 8px 0;
    }
    
    /* Primary button with gradient */
    .stButton > button {
        background: var(--accent-gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.6);
        filter: brightness(1.1);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Download button */
    .download-btn {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
        color: white;
        padding: 12px 24px;
        text-decoration: none;
        border-radius: 10px;
        display: inline-block;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6);
        filter: brightness(1.1);
        color: white;
        text-decoration: none;
    }
    
    /* Progress bar with gradient */
    .stProgress > div > div > div > div {
        background: var(--accent-gradient);
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(51, 65, 85, 0.5);
        border-radius: 12px;
        font-weight: 500;
        color: var(--text-primary);
        border: 1px solid var(--border);
    }
    
    /* Metric boxes with glass effect */
    .metric-box {
        background: rgba(51, 65, 85, 0.4);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        background: rgba(71, 85, 105, 0.5);
        border-color: var(--accent-primary);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(30, 41, 59, 0.6);
        border: 2px dashed rgba(6, 182, 212, 0.3);
        border-radius: 16px;
        color: var(--text-secondary);
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--accent-primary);
        background: rgba(30, 41, 59, 0.8);
    }
    
    /* Divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid var(--border);
    }
    
    .css-1d391kg .stMarkdown h3 {
        color: var(--accent-primary);
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 1.5rem;
    }
    
    .css-1d391kg .stMarkdown {
        color: var(--text-secondary);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(51, 65, 85, 0.6);
        border: 1px solid var(--border);
        border-radius: 10px;
        color: var(--text-primary);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text-primary);
    }
    
    /* Video container */
    .stVideo {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Status messages */
    .stInfo {
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 12px;
        color: var(--accent-primary);
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        color: #34d399;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        color: #f87171;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 12px;
        color: #fbbf24;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-primary);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-card);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--bg-hover);
    }
    
    /* Balloons animation enhancement */
    .stBalloons {
        filter: hue-rotate(180deg) brightness(1.2);
    }
    
    /* Section headers */
    h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Strong text in info boxes */
    .info-box strong {
        color: var(--accent-primary);
    }
    
    /* Preview sections */
    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 16px;
        border: 1px solid var(--border);
    }
</style>
""",
    unsafe_allow_html=True,
)


def extract_audio_from_video(video_path, audio_output_path):
    """Pull audio out of video file"""
    try:
        video = VideoFileClip(video_path)
        if video.audio:
            video.audio.write_audiofile(
                audio_output_path, codec="pcm_s16le", logger=None
            )
            video.close()
            return True
        else:
            st.error("❌ No audio track found in video")
            return False
    except Exception as e:
        st.error(f"❌ Audio extraction failed: {str(e)}")
        return False


def transcribe_audio_whisper(audio_path, model_name="base"):
    """Get text from audio using Whisper"""
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path, task="transcribe")
        detected_lang = result.get("language", "unknown")
        st.info(f"🌐 Detected language: **{detected_lang.upper()}**")
        return result
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")
        return None


def generate_srt_from_transcription(transcription):
    """Turn transcription into subtitle format"""
    try:
        subtitles = []
        for i, segment in enumerate(transcription.get("segments", [])):
            subtitle = srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=segment["start"]),
                end=timedelta(seconds=segment["end"]),
                content=segment["text"].strip(),
            )
            subtitles.append(subtitle)
        return srt.compose(subtitles)
    except Exception as e:
        st.error(f"❌ SRT generation failed: {str(e)}")
        return ""


def get_file_download_link(file_path, file_name, label):
    """Make a download button for files"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" class="download-btn">{label}</a>'


def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Video Caption Maker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Add professional captions to your videos using AI</p>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 About")
        st.markdown("""
        <div style="color: #94a3b8; line-height: 1.6;">
        This tool uses OpenAI's Whisper AI to automatically generate and add captions to your videos.
        
        <br><br>
        <strong style="color: #06b6d4;">How it works:</strong><br>
        1. Extracts audio from your video<br>
        2. Transcribes speech to text<br>
        3. Creates timed subtitles<br>
        4. Burns captions into video
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        model_option = st.selectbox(
            "Whisper Model",
            ["base", "small", "medium"],
            index=0,
            help="Larger models are more accurate but slower"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Requirements")
        st.markdown("""
        <div style="color: #94a3b8; font-size: 0.9rem;">
        • <strong style="color: #06b6d4;">Formats:</strong> MP4, MOV, AVI, MKV<br>
        • <strong style="color: #06b6d4;">Max size:</strong> 200MB<br>
        • <strong style="color: #06b6d4;">Languages:</strong> 99+ supported
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv", "MP4", "MOV", "AVI", "MKV"],
            help="MP4, MOV, AVI, or MKV format (max 200MB)"
        )

    with col2:
        st.markdown("### 📝 Process")
        steps = ["Upload Video", "Extract Audio", "AI Transcription", "Generate SRT", "Add Captions"]
        for i, step in enumerate(steps, 1):
            st.markdown(f"""
            <div class="step-item">
                <div class="step-number">{i}</div>
                <div class="step-text">{step}</div>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file is not None:
        # Create temp directory
        temp_dir = "temp_processing"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save uploaded file
            video_path = os.path.join(temp_dir, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Video preview
            with st.expander("🎥 Video Preview", expanded=True):
                st.video(uploaded_file)

            # Process button
            if st.button("🚀 Generate Captions", type="primary", use_container_width=True):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Extract audio
                status_text.info("📀 Step 1/5: Extracting audio from video...")
                audio_path = os.path.join(temp_dir, "audio.wav")
                
                if extract_audio_from_video(video_path, audio_path):
                    progress_bar.progress(20)
                    st.success("✅ Audio extracted")
                else:
                    st.stop()

                # Step 2: Transcribe
                status_text.info("🎤 Step 2/5: Transcribing audio with AI...")
                with st.spinner("This may take a minute..."):
                    transcription = transcribe_audio_whisper(audio_path, model_option)

                if transcription:
                    progress_bar.progress(40)
                    st.success("✅ Transcription complete")
                    
                    # Show transcription
                    with st.expander("📝 Transcription Preview"):
                        st.write(transcription["text"])
                else:
                    st.stop()

                # Step 3: Generate SRT
                status_text.info("📝 Step 3/5: Generating subtitle file...")
                srt_content = generate_srt_from_transcription(transcription)
                srt_path = os.path.join(temp_dir, "subtitles.srt")

                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)

                progress_bar.progress(60)
                st.success("✅ Subtitle file created")

                # Show subtitle preview
                with st.expander("📄 Subtitle Preview"):
                    subtitles = list(srt.parse(srt_content))[:5]
                    for sub in subtitles:
                        st.markdown(f"**{sub.start} → {sub.end}**")
                        st.write(f"_{sub.content}_")
                        st.markdown("---")

                # Step 4: Add captions to video
                status_text.info("🎬 Step 4/5: Adding captions to video...")
                output_video_path = os.path.join(temp_dir, "output_with_captions.mp4")

                try:
                    # Load video
                    video = VideoFileClip(video_path)
                    
                    # Create text clips for each subtitle
                    text_clips = []
                    for sub in list(srt.parse(srt_content)):
                        start_time = sub.start.total_seconds()
                        duration = sub.end.total_seconds() - start_time
                        
                        # Create text clip
                        txt_clip = TextClip(
                            text=sub.content,
                            font_size=24,
                            color='white',
                            font='Arial',
                            stroke_color='black',
                            stroke_width=1.5,
                            method='caption',
                            size=(video.w * 0.9, None)
                        ).with_duration(duration).with_start(start_time)
                        
                        # Position at bottom center
                        txt_clip = txt_clip.with_position(('center', 'bottom'))
                        text_clips.append(txt_clip)
                    
                    # Composite video with captions
                    if text_clips:
                        final_video = CompositeVideoClip([video] + text_clips)
                    else:
                        final_video = video
                    
                    # Write output
                    final_video.write_videofile(
                        output_video_path,
                        codec='libx264',
                        audio_codec='aac',
                        fps=video.fps,
                        logger=None,
                        threads=4
                    )
                    
                    # Cleanup
                    video.close()
                    final_video.close()
                    
                    progress_bar.progress(80)
                    st.success("✅ Captions added to video")
                    
                except Exception as e:
                    st.error(f"Failed to add captions: {str(e)}")
                    st.warning("Saving original video without captions")
                    import shutil
                    shutil.copy(video_path, output_video_path)

                # Step 5: Complete
                status_text.success("✨ Step 5/5: Processing complete!")
                progress_bar.progress(100)
                time.sleep(0.5)
                status_text.empty()

                # Results section
                st.markdown("---")
                st.markdown("## 🎉 Your Video is Ready!")
                
                # Results in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📹 Preview")
                    st.video(output_video_path)
                
                with col2:
                    st.markdown("### 📄 Subtitles")
                    download_link = get_file_download_link(
                        srt_path, 
                        "subtitles.srt", 
                        "Download SRT File"
                    )
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                    # Show SRT content
                    with open(srt_path, "r") as f:
                        srt_text = f.read()
                    st.text_area("SRT Preview", srt_text[:500] + "...", height=150)
                
                with col3:
                    st.markdown("### 🎬 Final Video")
                    download_link = get_file_download_link(
                        output_video_path,
                        "video_with_captions.mp4",
                        "Download Video"
                    )
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                    # Stats
                    st.markdown("### 📊 Statistics")
                    video_clip = VideoFileClip(video_path)
                    duration = video_clip.duration
                    video_clip.close()
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{duration:.0f}s</div>
                            <div class="metric-label">Duration</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        word_count = len(transcription["text"].split())
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value">{word_count}</div>
                            <div class="metric-label">Words</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    segment_count = len(transcription.get("segments", []))
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{segment_count}</div>
                        <div class="metric-label">Captions</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Success message
                st.markdown("""
                <div class="success-box">
                    🎬 Success! Your video has been processed with captions. Download your files above.
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Cleanup can be added here if needed
            pass

    else:
        # Welcome message when no video uploaded
        st.markdown("""
        <div class="info-box">
            <h3>🎬 Welcome to Video Caption Maker</h3>
            <p>Create accessible, engaging videos with AI-powered captions in minutes.</p>
            
            <h4>✨ Features</h4>
            <ul>
                <li>🤖 Automatic speech recognition using OpenAI's Whisper AI</li>
                <li>🌐 Supports 99+ languages</li>
                <li>⚡ Fast and accurate transcription</li>
                <li>🎨 Clean, professional captions</li>
                <li>📥 Download video with burned-in captions</li>
            </ul>
            
            <h4>🚀 Quick Start</h4>
            <ol>
                <li>Upload your video using the button above</li>
                <li>Click "Generate Captions"</li>
                <li>Wait for AI processing</li>
                <li>Download your captioned video</li>
            </ol>
            
            <p><strong>Ready to get started? Upload a video file to begin!</strong></p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()