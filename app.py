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
st.set_page_config(page_title="Video Caption Maker", page_icon="🎬", layout="wide")

# Some basic styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def extract_audio_from_video(video_path, audio_output_path):
    """Pull audio out of video file"""
    try:
        video = VideoFileClip(video_path)
        if video.audio:
            # Save as WAV for Whisper
            video.audio.write_audiofile(
                audio_output_path, codec="pcm_s16le", logger=None
            )
            video.close()
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Couldn't extract audio: {str(e)}")
        return False


def transcribe_audio_whisper(audio_path, model_name="base"):
    """Get text from audio using Whisper"""
    try:
        # Using base model for good balance of speed and accuracy
        model = whisper.load_model(model_name)

        # Let Whisper figure out what language it is
        result = model.transcribe(audio_path, task="transcribe")

        # Show user what language we detected
        detected_lang = result.get("language", "unknown")
        st.info(f"🌐 Found language: **{detected_lang}**")

        return result
    except Exception as e:
        st.error(f"Transcription didn't work: {str(e)}")
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
        st.error(f"Oops, something went wrong with SRT: {str(e)}")
        return ""


def burn_subtitles_onto_video(video_path, srt_content, output_path):
    """Add subtitles directly to video"""
    try:
        # Parse the subtitle file
        subtitles = list(srt.parse(srt_content))

        # Load the video
        video = VideoFileClip(video_path)

        # Create all subtitle clips
        subtitle_clips = []

        for sub in subtitles:
            start_sec = sub.start.total_seconds()
            end_sec = sub.end.total_seconds()
            duration = end_sec - start_sec

            # Make text clip for this subtitle
            txt_clip = TextClip(
                sub.content,
                fontsize=28,
                color="white",
                font="Arial",
                stroke_color="black",
                stroke_width=2,
                size=(video.w * 0.9, None),
                method="caption",
            ).with_duration(duration)

            # Put it at bottom center
            txt_clip = txt_clip.with_position(("center", "bottom")).with_start(
                start_sec
            )
            subtitle_clips.append(txt_clip)

        # Combine video and subtitles
        if subtitle_clips:
            final_video = CompositeVideoClip([video] + subtitle_clips)
        else:
            final_video = video

        # Save the result
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            logger=None,
            threads=4,
        )

        # Clean up
        video.close()
        if "final_video" in locals():
            final_video.close()

        return True
    except Exception as e:
        st.error(f"Had trouble burning subtitles: {str(e)}")
        return False


def get_file_download_link(file_path, file_name, label):
    """Make a download button for files"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" style="background-color: #FF4B4B; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">{label}</a>'


def main():
    # Big title
    st.markdown(
        '<h1 class="main-header">🎬 AI Video Caption Generator</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar with info
    with st.sidebar:
        st.markdown("### 📋 About")
        st.markdown("""
        Just upload and let it do its thing.
        This tool:
        1. Grabs audio from your video
        2. Transcribes it using Whisper
        3. Creates subtitle file
        4. Puts subtitles on video
        5. Works with 99 languages
        
        **Supported formats:** MP4, MOV, AVI, MKV
        **Max file size:** 200MB
        """)

        st.markdown("### ⚙️ Settings")
        model_option = st.selectbox(
            "Pick Whisper Model",
            ["base (Fast & Good)", "small (Better)", "medium (Best)"],
            index=0,
        )

        st.markdown("### 🎯 Features")
        st.markdown("""
        - No extra setup needed
        - Pretty fast
        - Good accuracy
        - Works with many languages
        - Download everything
        """)

    # Two columns layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Pick a video file",
            type=["mp4", "mov", "avi", "mkv", "MP4", "MOV", "AVI", "MKV"],
            help="Select the video you want to add captions to",
        )

    with col2:
        st.markdown("### 📊 Steps")
        steps = [
            "1. Upload",
            "2. Extract Audio",
            "3. Transcribe",
            "4. Make SRT",
            "5. Add Subtitles",
        ]
        for step in steps:
            st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Make a temp folder for processing
        temp_dir = "temp_processing"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save the uploaded file
            video_path = os.path.join(temp_dir, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Show preview
            st.markdown("### 👁️ Preview")
            st.video(uploaded_file)

            # Main processing button
            if st.button(
                "🚀 Start Processing", type="primary", use_container_width=True
            ):
                # Setup progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Get audio
                status_text.text("Step 1/5: Getting audio from video...")
                audio_path = os.path.join(temp_dir, "audio.wav")
                if extract_audio_from_video(video_path, audio_path):
                    progress_bar.progress(20)
                    st.success("✓ Audio extracted")
                else:
                    st.error("Couldn't extract audio")
                    return

                # Step 2: Transcribe
                status_text.text("Step 2/5: Converting speech to text...")
                with st.spinner("This might take a minute..."):
                    model_name = model_option.split()[0]
                    transcription = transcribe_audio_whisper(audio_path, model_name)

                if transcription:
                    progress_bar.progress(40)
                    st.success("✓ Transcription done")

                    # Show what we got
                    with st.expander("📝 What was said", expanded=True):
                        st.write(transcription["text"])
                else:
                    st.error("Transcription failed")
                    return

                # Step 3: Create subtitle file
                status_text.text("Step 3/5: Building subtitle file...")
                srt_content = generate_srt_from_transcription(transcription)
                srt_path = os.path.join(temp_dir, "subtitles.srt")

                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)

                progress_bar.progress(60)
                st.success("✓ SRT file ready")

                # Show preview of subtitles
                with st.expander("📄 Subtitle preview"):
                    subtitles = list(srt.parse(srt_content))[:5]  # Just first few
                    for sub in subtitles:
                        st.markdown(f"**{sub.start} → {sub.end}**")
                        st.write(sub.content)
                        st.markdown("---")

                # Step 4: Add subtitles to video
                status_text.text("Step 4/5: Putting subtitles on video...")
                output_video_path = os.path.join(temp_dir, "output_with_subtitles.mp4")

                # Try the main method first
                try:
                    video = VideoFileClip(video_path)
                    clips = [video]

                    # Add each subtitle as a text overlay
                    for sub in list(srt.parse(srt_content)):
                        start_sec = sub.start.total_seconds()
                        duration = sub.end.total_seconds() - start_sec

                        # Make the text clip
                        txt_clip = (
                            TextClip(text=sub.content, font_size=18, color="white")
                            .with_duration(duration)
                            .with_start(start_sec)
                        )

                        txt_clip = txt_clip.with_position(("center", "bottom"))
                        clips.append(txt_clip)

                    # Combine everything
                    final_video = CompositeVideoClip(clips)
                    final_video.write_videofile(
                        output_video_path,
                        fps=video.fps,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None,
                    )

                    # Clean up
                    video.close()
                    final_video.close()

                    progress_bar.progress(80)
                    st.success("✓ Subtitles added")

                except Exception as e:
                    st.warning(f"Had to use backup method: {str(e)}")
                    # Simple fallback - just copy original
                    try:
                        import shutil

                        shutil.copy(video_path, output_video_path)
                        st.info(
                            "Saved video without subtitles (but SRT is still available)"
                        )
                    except:
                        st.error("Something went wrong with video processing")
                        return

                # Step 5: Wrap up
                status_text.text("Step 5/5: Wrapping up...")
                progress_bar.progress(100)
                time.sleep(0.5)
                status_text.text("✓ All done!")

                # Show results
                st.markdown("### 🎉 Here's what we got")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 📹 Video with captions")
                    try:
                        st.video(output_video_path)
                    except:
                        st.warning("Can't preview video here")

                with col2:
                    st.markdown("#### 📄 Subtitle file")
                    # SRT download button
                    srt_download = get_file_download_link(
                        srt_path, "subtitles.srt", "📥 Get SRT"
                    )
                    st.markdown(srt_download, unsafe_allow_html=True)

                    # Show file content
                    with open(srt_path, "r") as f:
                        st.text_area("SRT preview", f.read(), height=200)

                with col3:
                    st.markdown("#### 🎬 Final video")
                    # Video download
                    if os.path.exists(output_video_path):
                        video_download = get_file_download_link(
                            output_video_path,
                            "video_with_subtitles.mp4",
                            "📥 Download video",
                        )
                        st.markdown(video_download, unsafe_allow_html=True)

                    # Some stats
                    st.markdown("##### 📊 Quick stats")
                    video_duration = 0
                    try:
                        video_clip = VideoFileClip(video_path)
                        video_duration = video_clip.duration
                        video_clip.close()
                    except:
                        pass

                    st.markdown(f"""
                    - **Video length:** {video_duration:.1f} seconds
                    - **Words:** {len(transcription["text"].split())}
                    - **Subtitle lines:** {len(transcription.get("segments", []))}
                    - **Model:** {model_option.split()[0]}
                    """)

                # Celebration time
                st.balloons()
                st.markdown(
                    '<div class="success-box">🎉 Done! Grab your files above.</div>',
                    unsafe_allow_html=True,
                )

        finally:
            # Leave files for download, cleanup happens elsewhere
            pass

    else:
        # Show instructions when nothing uploaded
        st.markdown(
            """
        <div class="info-box">
        <h3>📋 How to use:</h3>
        <ol>
            <li>Upload a video using the button on the left</li>
            <li>Hit "Start Processing"</li>
            <li>Wait a few minutes while AI does its thing</li>
            <li>Download your video with subtitles</li>
        </ol>
        
        <h3>✨ What it does:</h3>
        <ul>
            <li>Listens to audio and writes it down</li>
            <li>Makes sure timing is right</li>
            <li>Adds nice-looking captions</li>
            <li>Works right in your browser</li>
            <li>Free to use</li>
        </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
