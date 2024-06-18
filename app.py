import streamlit as st
from transformers import pipeline
from pydub import AudioSegment
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Initialize the pipeline
pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# CSS for styling
st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        background-color: transparent;
    }
    .stButton>button {
        background-color: #4682B4;
        color: white;
        font-weight: bold;
        padding: 0.75rem 1.25rem;
        border-radius: 0.5rem;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page layout
st.markdown(
    """
    <div class="centered">
        <h1 style="font-size: 3rem; margin-bottom: 2rem;">Speech Emotion Recognition</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Function to classify the emotion
def classify_emotion(audio_path):
    result = pipe(audio_path)
    return result

# Option to upload an audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Option to record audio
st.write("Or record an audio sample")
record = st.button("Record")

# Process the uploaded file
if uploaded_file is not None:
    audio = AudioSegment.from_file(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio.export(temp_audio_file.name, format="wav")
        result = classify_emotion(temp_audio_file.name)
        labels = [r['label'] for r in result]
        scores = [r['score'] for r in result]
        plt.figure(figsize=(10, 5))
        plt.barh(labels, scores, color='#4682B4')
        plt.xlabel('Scores')
        plt.title('Emotion Classification')
        st.pyplot(plt)

# Process the recorded audio
if record:
    class AudioProcessor(AudioProcessorBase):
        def recv(self, frame):
            return frame

    webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDONLY, audio_processor_factory=AudioProcessor, media_stream_constraints={"audio": True, "video": False})

    if webrtc_ctx.state.playing:
        st.write("Recording...")
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            audio = audio_frames[0].to_ndarray()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                audio_segment = AudioSegment(audio.tobytes(), frame_rate=webrtc_ctx.audio_receiver.sample_rate, sample_width=audio.dtype.itemsize, channels=1)
                audio_segment.export(temp_audio_file.name, format="wav")
                result = classify_emotion(temp_audio_file.name)
                labels = [r['label'] for r in result]
                scores = [r['score'] for r in result]
                plt.figure(figsize=(10, 5))
                plt.barh(labels, scores, color='#4682B4')
                plt.xlabel('Scores')
                plt.title('Emotion Classification')
                st.pyplot(plt)
