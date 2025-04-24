import streamlit as st
import numpy as np
from math import sqrt, sin, pi
import io
import sys
from pydub import AudioSegment

# Check for FFmpeg
try:
    AudioSegment.ffmpeg = "/usr/bin/ffmpeg"  # Default Linux path
    test_sound = AudioSegment.silent(duration=100)
except:
    st.warning("FFmpeg not found - audio generation may not work")

# Set page config
st.set_page_config(
    page_title="SoundTempo Putt Generator",
    page_icon="⛳",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stSlider [data-baseweb="slider"] {
        width: 95%;
    }
    .stAudio {
        border-radius: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        margin: 10px 0;
    }
    .stMarkdown h1 {
        color: #2e8b57;
    }
</style>
""", unsafe_allow_html=True)

class SoundTempoPuttGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05
        self.stimp_conversion = 0.3048 / 1.83
        self.club_length = 0.9
        self.gravity = 9.81

    def calculate_required_velocity(self, distance_ft, stimp, slope_percent):
        distance_m = distance_ft * 0.3048
        stimp_mps = stimp * self.stimp_conversion
        slope_factor = 1 + (slope_percent / 100)
        return sqrt(2 * self.gravity * 0.3 * stimp_mps * distance_m) * slope_factor

    def calculate_swing_parameters(self, core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent):
        dsi_time = 30 / core_tempo_bpm
        backswing_time = dsi_time * backswing_rhythm
        required_velocity = self.calculate_required_velocity(distance_ft, stimp, slope_percent)
        angular_velocity = required_velocity / (self.club_length * 0.7)
        backswing_length_m = angular_velocity * dsi_time * self.club_length * 1.2
        return {
            'dsi_time': dsi_time,
            'backswing_time': backswing_time,
            'backswing_length_in': backswing_length_m * 39.37,
            'required_velocity': required_velocity
        }

    def generate_impact_chirp(self):
        t = np.linspace(0, self.impact_chirp_duration, 
                       int(self.sample_rate * self.impact_chirp_duration), False)
        chirp = np.sin(2 * pi * (1000 + 2000 * t/self.impact_chirp_duration) * t)
        envelope = np.linspace(1, 0, len(t))
        return chirp * envelope

    def generate_tone(self, duration, start_freq, end_freq, volume_envelope):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        freq = start_freq + (end_freq - start_freq) * t/duration
        tone = np.sin(2 * pi * freq * t)
        return tone * volume_envelope

    def generate_putt_audio(self, core_tempo_bpm, backswing_rhythm, distance_ft, 
                          stimp=10, slope_percent=0, handedness='right'):
        params = self.calculate_swing_parameters(
            core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent)
        
        backswing_tone = self.generate_tone(
            params['backswing_time'], 400, 600,
            np.concatenate([
                np.linspace(0, 1, int(0.1 * params['backswing_time'] * self.sample_rate)),
                np.ones(int(0.8 * params['backswing_time'] * self.sample_rate)),
                np.linspace(1, 0, int(0.1 * params['backswing_time'] * self.sample_rate))
            ])
        )
        
        downswing_tone = self.generate_tone(
            params['dsi_time'], 600, 400,
            np.concatenate([
                np.linspace(0, 1, int(0.1 * params['dsi_time'] * self.sample_rate)),
                np.ones(int(0.7 * params['dsi_time'] * self.sample_rate)),
                np.linspace(1, 0, int(0.2 * params['dsi_time'] * self.sample_rate))
            ])
        )
        
        impact_chirp = self.generate_impact_chirp()
        backswing_beep = self.generate_tone(
            self.backswing_beep_duration, 1200, 1200, 
            np.linspace(1, 0, int(self.sample_rate * self.backswing_beep_duration)))
        
        if handedness == 'right':
            pan_backswing = np.linspace(1, 0, len(backswing_tone))
            pan_downswing = np.linspace(0, 1, len(downswing_tone))
        else:
            pan_backswing = np.linspace(0, 1, len(backswing_tone))
            pan_downswing = np.linspace(1, 0, len(downswing_tone))

        left_channel = np.concatenate([
            backswing_tone * pan_backswing,
            downswing_tone * pan_downswing
        ])
        right_channel = np.concatenate([
            backswing_tone * (1 - pan_backswing),
            downswing_tone * (1 - pan_downswing)
        ])

        beep_pos = int(0.9 * len(backswing_tone))
        left_channel[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep * 0.5
        right_channel[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep * 0.5

        impact_pos = len(backswing_tone)
        left_channel[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp * 0.7
        right_channel[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp * 0.7

        max_amplitude = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        left_channel /= max_amplitude
        right_channel /= max_amplitude

        return left_channel, right_channel, params

    def save_as_mp3(self, left_channel, right_channel):
        stereo_audio = np.column_stack((
            (left_channel * 32767).astype(np.int16),
            (right_channel * 32767).astype(np.int16)
        ))
        audio_segment = AudioSegment(
            stereo_audio.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=2
        )
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="mp3", bitrate="192k")
        return buffer

# Streamlit UI
st.title("⛳ SoundTempo Putt Generator")
st.markdown("""
Create customized putting tones based on green conditions and stroke mechanics.  
Adjust the sliders to match your putting scenario and generate audio feedback.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Stroke Parameters")
    core_tempo = st.slider("Core Tempo (BPM)", 72, 130, 90, 
                          help="Your natural downswing tempo (90bpm = 0.333s downswing)")
    rhythm = st.slider("Backswing Rhythm Ratio", 1.7, 2.5, 2.0, 0.1,
                      help="Backswing time relative to downswing (2.0 = twice as long)")
    handedness = st.radio("Handedness", ["Right", "Left"], index=0)

with col2:
    st.subheader("Green Conditions")
    distance = st.slider("Distance (feet)", 1, 60, 15)
    stimp = st.slider("Stimp Rating", 5.0, 16.0, 10.0, 0.5,
                     help="Green speed (higher = faster)")
    slope = st.slider("Slope (%)", -10.0, 10.0, 0.0, 0.5,
                     help="Negative = downhill, Positive = uphill")

# Generate audio
if st.button("Generate Putting Tone", type="primary"):
    generator = SoundTempoPuttGenerator()
    try:
        left, right, params = generator.generate_putt_audio(
            core_tempo_bpm=core_tempo,
            backswing_rhythm=rhythm,
            distance_ft=distance,
            stimp=stimp,
            slope_percent=slope,
            handedness=handedness.lower()
        )
        
        # Display results
        st.subheader("Stroke Analysis")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Backswing Time", f"{params['backswing_time']:.3f} seconds")
            st.metric("Downswing Time", f"{params['dsi_time']:.3f} seconds")
        with col4:
            st.metric("Backswing Length", f"{params['backswing_length_in']:.1f} inches")
            st.metric("Required Impact Velocity", f"{params['required_velocity']:.2f} m/s")
        
        try:
            audio_buffer = generator.save_as_mp3(left, right)
            st.audio(audio_buffer, format="audio/mp3")
            st.download_button(
                label="Download MP3",
                data=audio_buffer,
                file_name=f"putt_{distance}ft_stimp{stimp}_slope{slope}%.mp3",
                mime="audio/mp3"
            )
        except Exception as e:
            st.error(f"Audio export failed. FFmpeg may not be installed. Error: {str(e)}")
            st.warning("Try running locally with FFmpeg installed or check Streamlit Cloud logs")
            
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("""
Based on SoundTempo Putt™ technology - Golf training system for developing consistent putting rhythm.  
*Note: For best results, use headphones to hear the stereo panning effects.*
""")