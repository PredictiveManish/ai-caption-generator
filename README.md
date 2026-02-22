# 🎬 AI Video Caption Generator

A simple web app that automatically generates subtitles for videos using AI.

Built with **Streamlit**, **OpenAI Whisper**, and **MoviePy**.

---

## 🚀 Features

* 📤 Upload video files (MP4, MOV, AVI, MKV)
* 🎧 Extract audio from video
* 🧠 Transcribe speech using Whisper (multilingual support)
* 📝 Generate subtitles in SRT format
* 🎬 Burn subtitles directly onto video
* 📥 Download SRT file and final video

---

## 🧠 How It Works

1. Upload a video
2. Audio is extracted using MoviePy
3. Whisper model converts speech → text
4. Transcription is converted into SRT subtitles
5. Subtitles are added to video
6. Download results

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (UI)
* **Whisper** (speech-to-text)
* **MoviePy** (video processing)
* **SRT** (subtitle formatting)
* **Pydub** (audio handling)

---

## 📦 Installation

```bash
pip install -r requirements.txt

(Better is to create a virtual environment before this command either using venv or conda)
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Supported Formats

* Video: MP4, MOV, AVI, MKV
* Output:

  * `.mp4` (video with subtitles)
  * `.srt` (subtitle file)

---

## ⚠️ Notes

* Processing time depends on video length and model size
* Whisper runs locally (no API required)
* Large videos may take longer to process

---

## 🔮 Future Improvements

* Better subtitle styling (fonts, colors)
* GPU acceleration for faster processing
* Real-time captioning
* Multi-language subtitle export

---

## 🙌 Acknowledgements

* OpenAI Whisper for speech recognition
* MoviePy for video processing
* Streamlit for UI framework

---

## 📌 Author

Manish Tiwari
(Built as part of learning AI & real-world applications)
