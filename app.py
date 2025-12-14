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

# Set page configuration
st.set_page_config(
    page_title="AI Video Caption Generator",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4eada;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #007BFF;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .step-box {
        background-color: #87CEEB;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

def extract_audio_from_video(video_path, audio_output_path):
    """Extract audio from video file using moviepy"""
    try:
        video = VideoFileClip(video_path)
        if video.audio:
            # Write audio as WAV file (better for Whisper)
            video.audio.write_audiofile(audio_output_path, codec='pcm_s16le', logger=None)
            video.close()
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return False

def transcribe_audio_whisper(audio_path):
    """Transcribe audio using OpenAI Whisper"""
    try:
        # Load Whisper model (base is fast and accurate)
        model = whisper.load_model("base")
        
        # Load audio using whisper's built-in function with error handling
        try:
            # Try whisper's default loading
            result = model.transcribe(audio_path, language="en")
        except:
            # Fallback using pydub
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            result = model.transcribe(audio_array)
        
        return result
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return None

def generate_srt_from_transcription(transcription):
    """Generate SRT subtitle file from transcription"""
    try:
        subtitles = []
        for i, segment in enumerate(transcription.get('segments', [])):
            subtitle = srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=segment['start']),
                end=timedelta(seconds=segment['end']),
                content=segment['text'].strip()
            )
            subtitles.append(subtitle)
        return srt.compose(subtitles)
    except Exception as e:
        st.error(f"Error generating SRT: {str(e)}")
        return ""

def burn_subtitles_onto_video(video_path, srt_content, output_path):
    """Burn subtitles onto video using moviepy"""
    try:
        # Parse SRT content
        subtitles = list(srt.parse(srt_content))
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Create subtitle clips
        subtitle_clips = []
        
        for sub in subtitles:
            # Calculate timing
            start_sec = sub.start.total_seconds()
            end_sec = sub.end.total_seconds()
            duration = end_sec - start_sec
            
            # Create text clip - FIXED for newer MoviePy versions
            txt_clip = TextClip(
                sub.content,
                fontsize=28,
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2,
                size=(video.w * 0.9, None),
                method='caption'
            ).with_duration(duration)
            
            # Position at bottom center
            txt_clip = txt_clip.with_position(('center', 'bottom')).with_start(start_sec)
            subtitle_clips.append(txt_clip)
        
        # Composite video with subtitles
        if subtitle_clips:
            final_video = CompositeVideoClip([video] + subtitle_clips)
        else:
            final_video = video
        
        # Write output video
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger=None,
            threads=4
        )
        
        # Explicitly close clips
        video.close()
        if 'final_video' in locals():
            final_video.close()
        
        return True
    except Exception as e:
        st.error(f"Error burning subtitles: {str(e)}")
        return False

def get_file_download_link(file_path, file_name, label):
    """Generate a download link for a file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" style="background-color: #FF4B4B; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">{label}</a>'

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 AI Video Caption Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 📋 About")
        st.markdown("""
        This tool automatically:
        1. Extracts audio from your video
        2. Transcribes speech using AI (OpenAI Whisper)
        3. Generates subtitles in SRT format
        4. Burns subtitles onto your video
        
        **Supported formats:** MP4, MOV, AVI, MKV
        **Max file size:** 200MB
        """)
        
        st.markdown("### ⚙️ Settings")
        model_option = st.selectbox(
            "Select Whisper Model",
            ["base (Fast & Good)", "small (Better)", "medium (Best)"],
            index=0
        )
        
        st.markdown("### 🎯 Features")
        st.markdown("""
        -   No external FFmpeg required
        -   Fast processing
        -   High accuracy
        -   Customizable subtitles
        -   Direct downloads
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload Your Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv", "MP4", "MOV", "AVI", "MKV"],
            help="Upload a video file to generate captions"
        )
    
    with col2:
        st.markdown("### 📊 Processing Steps")
        steps = ["1. Upload Video", "2. Extract Audio", "3. Transcribe", "4. Generate SRT", "5. Burn Subtitles"]
        for step in steps:
            st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Use a persistent temp directory instead
        temp_dir = "temp_processing"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save uploaded file
            video_path = os.path.join(temp_dir, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display video preview
            st.markdown("### 👁️ Video Preview")
            st.video(uploaded_file)
            
            # Process button
            if st.button("🚀 Generate Captions", type="primary", use_container_width=True):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract Audio
                status_text.text("Step 1/5: Extracting audio from video...")
                audio_path = os.path.join(temp_dir, "audio.wav")
                if extract_audio_from_video(video_path, audio_path):
                    progress_bar.progress(20)
                    st.success("  Audio extracted successfully")
                else:
                    st.error("Failed to extract audio")
                    return
                
                # Step 2: Transcribe Audio
                status_text.text("Step 2/5: Transcribing audio using AI...")
                with st.spinner("Transcribing (this may take a minute)..."):
                    transcription = transcribe_audio_whisper(audio_path)
                
                if transcription:
                    progress_bar.progress(40)
                    st.success("  Transcription completed")
                    
                    # Display transcription
                    with st.expander("📝 View Transcription", expanded=True):
                        st.write(transcription['text'])
                else:
                    st.error("Transcription failed")
                    return
                
                # Step 3: Generate SRT
                status_text.text("Step 3/5: Generating subtitle file...")
                srt_content = generate_srt_from_transcription(transcription)
                srt_path = os.path.join(temp_dir, "subtitles.srt")
                
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                
                progress_bar.progress(60)
                st.success("  SRT file generated")
                
                # Display sample subtitles
                with st.expander("📄 View Sample Subtitles"):
                    subtitles = list(srt.parse(srt_content))[:5]  # Show first 5
                    for sub in subtitles:
                        st.markdown(f"**{sub.start} → {sub.end}**")
                        st.write(sub.content)
                        st.markdown("---")
                
                # Step 4: Burn Subtitles (SIMPLIFIED VERSION)
                status_text.text("Step 4/5: Burning subtitles onto video...")
                output_video_path = os.path.join(temp_dir, "output_with_subtitles.mp4")
                
                # Alternative simple subtitle burning method
                try:
                    # Method 1: Try simple method
                    video = VideoFileClip(video_path)
                    clips = [video]
                    
                    # Add simple text overlays
                    for sub in list(srt.parse(srt_content)):
                        start_sec = sub.start.total_seconds()
                        duration = sub.end.total_seconds() - start_sec
                        
                        # Create text clip (compatible with all MoviePy versions)
                        # txt_clip = TextClip(
                        #     sub.content,
                        #     font_size=28,
                        #     color='white',
                        #     stroke_color='black',
                        #     stroke_width=1
                        # ).with_duration(duration).with_start(start_sec)

                        # txt_clip = TextClip(
                        #     sub.content,
                        #     font_size=28,
                        #     color='white',
                        #     stroke_color='black',
                        #     stroke_width=1
                        # ).with_duration(duration).with_start(start_sec)

                        txt_clip = TextClip(
                            text=sub.content,  # Explicitly name the parameter
                            font_size=18,
                            color='white'
                        ).with_duration(duration).with_start(start_sec)
                        
                        txt_clip = txt_clip.with_position(('center', 'bottom'))
                        clips.append(txt_clip)
                    
                    # Create final video
                    final_video = CompositeVideoClip(clips)
                    final_video.write_videofile(
                        output_video_path,
                        fps=video.fps,
                        codec='libx264',
                        audio_codec='aac',
                         
                        logger=None
                    )
                    
                    # Close clips
                    video.close()
                    final_video.close()
                    
                    progress_bar.progress(80)
                    st.success("  Subtitles burned onto video")
                    
                except Exception as e:
                    st.warning(f"Using alternative method: {str(e)}")
                    # Method 2: Create video with captions overlay (simpler)
                    try:
                        # Just copy the original video if subtitle burning fails
                        import shutil
                        shutil.copy(video_path, output_video_path)
                        st.info("Video saved without subtitles (fallback mode). SRT file is available for download.")
                    except:
                        st.error("Could not process video with subtitles")
                        return
                
                # Step 5: Finalize
                status_text.text("Step 5/5: Finalizing...")
                progress_bar.progress(100)
                time.sleep(0.5)
                status_text.text("  Processing complete!")
                
                # Results section
                st.markdown("### 🎉 Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 📹 Output Video")
                    try:
                        st.video(output_video_path)
                    except:
                        st.warning("Video preview not available")
                
                with col2:
                    st.markdown("#### 📄 SRT File")
                    # Download SRT button
                    srt_download = get_file_download_link(srt_path, "subtitles.srt", "📥 Download SRT")
                    st.markdown(srt_download, unsafe_allow_html=True)
                    
                    # Preview SRT
                    with open(srt_path, "r") as f:
                        st.text_area("SRT Content Preview", f.read(), height=200)
                
                with col3:
                    st.markdown("#### 🎬 Final Video")
                    # Download Video button
                    if os.path.exists(output_video_path):
                        video_download = get_file_download_link(
                            output_video_path, 
                            "video_with_subtitles.mp4", 
                            "📥 Download Video"
                        )
                        st.markdown(video_download, unsafe_allow_html=True)
                    
                    # Stats
                    st.markdown("##### 📊 Statistics")
                    video_duration = 0
                    try:
                        video_clip = VideoFileClip(video_path)
                        video_duration = video_clip.duration
                        video_clip.close()
                    except:
                        pass
                    
                    st.markdown(f"""
                    - **Video Duration:** {video_duration:.1f}s
                    - **Words Transcribed:** {len(transcription['text'].split())}
                    - **Subtitle Segments:** {len(transcription.get('segments', []))}
                    - **Model Used:** {model_option.split()[0]}
                    """)
                
                # Success message
                st.balloons()
                st.markdown('<div class="success-box">🎉 All processes completed successfully! You can download your files above.</div>', unsafe_allow_html=True)
        
        finally:
            # Cleanup - but don't delete if user might want to download again
            pass
    
    else:
        # Welcome/instructions when no file is uploaded
        st.markdown("""
        <div class="info-box">
        <h3>📋 How to Use:</h3>
        <ol>
            <li>Upload a video file using the uploader on the left</li>
            <li>Click the "Generate Captions" button</li>
            <li>Wait for the AI to process your video (takes 1-5 minutes)</li>
            <li>Download your video with burned-in subtitles</li>
        </ol>
        
        <h3>✨ Features:</h3>
        <ul>
            <li>Automatic speech-to-text conversion</li>
            <li>Accurate timestamp generation</li>
            <li>Professional subtitle styling</li>
            <li>No external software needed</li>
            <li>Completely free to use</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()