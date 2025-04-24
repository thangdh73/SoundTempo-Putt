import streamlit as st
import numpy as np
from math import sqrt, sin, pi
import io
import sys
import os
from pydub import AudioSegment

# Check for FFmpeg
try:
    if sys.platform == "win32":
        # Try common Windows FFmpeg paths
        ffmpeg_paths = [
            os.path.join(os.environ.get("PROGRAMFILES", ""), "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "ffmpeg", "bin", "ffmpeg.exe"),
            "ffmpeg.exe"  # If it's in PATH
        ]
        for path in ffmpeg_paths:
            if os.path.exists(path):
                AudioSegment.converter = path
                break
    else:
        AudioSegment.converter = "/usr/bin/ffmpeg"  # Default Linux path
    
    # Test FFmpeg
    test_sound = AudioSegment.silent(duration=100)
except Exception as e:
    st.warning(f"FFmpeg check failed: {str(e)}. Audio may not work properly. Please ensure FFmpeg is installed and accessible.")

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
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.audio_segments = []
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

    def generate_tone(self, frequency, duration, volume=0.5):
        """Generate a sine wave tone with proper sample alignment"""
        try:
            t = np.linspace(0, duration, int(duration * self.sample_rate), False)
            samples = np.sin(2 * np.pi * frequency * t) * volume
            samples = np.clip(samples, -1, 1)  # Ensure samples are within valid range
            return samples
        except Exception as e:
            st.error(f"Error generating tone: {str(e)}")
            return np.zeros(int(duration * self.sample_rate))
            
    def add_audio_cue(self, frequency, duration, volume=0.5):
        """Add an audio cue with proper error handling"""
        try:
            samples = self.generate_tone(frequency, duration, volume)
            # Convert to 16-bit PCM
            samples = (samples * 32767).astype(np.int16)
            # Create AudioSegment
            audio_segment = AudioSegment(
                samples.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=1
            )
            self.audio_segments.append(audio_segment)
        except Exception as e:
            st.error(f"Error adding audio cue: {str(e)}")
            
    def generate_putting_sequence(self, distance, stimp, tempo=60):
        """Generate a complete putting sequence with proper error handling"""
        try:
            # Calculate swing parameters
            swing_time = self.calculate_swing_time(distance, stimp)
            backswing_time = swing_time * 0.4
            follow_through_time = swing_time * 0.6
            
            # Generate backswing beep
            self.add_audio_cue(880, 0.1, 0.3)  # A5 note
            
            # Generate impact chirp
            self.add_audio_cue(440, 0.05, 0.5)  # A4 note
            
            # Generate follow-through beep
            self.add_audio_cue(220, 0.1, 0.2)  # A3 note
            
            # Combine all segments
            if self.audio_segments:
                final_audio = self.audio_segments[0]
                for segment in self.audio_segments[1:]:
                    final_audio = final_audio + segment
                return final_audio
            else:
                return AudioSegment.silent(duration=1000)
        except Exception as e:
            st.error(f"Error generating putting sequence: {str(e)}")
            return AudioSegment.silent(duration=1000)
            
    def save_audio(self, audio, filename):
        """Save audio with proper error handling"""
        try:
            if isinstance(audio, AudioSegment):
                audio.export(filename, format="mp3")
                st.success(f"Audio saved successfully as {filename}")
            else:
                st.error("Invalid audio format")
        except Exception as e:
            st.error(f"Error saving audio: {str(e)}")

    def generate_impact_chirp(self):
        t = np.linspace(0, self.impact_chirp_duration, 
                       int(self.sample_rate * self.impact_chirp_duration), False)
        chirp = np.sin(2 * pi * (1000 + 2000 * t/self.impact_chirp_duration) * t)
        envelope = np.linspace(1, 0, len(t))
        return chirp * envelope

    def generate_putt_audio(self, core_tempo_bpm, backswing_rhythm, distance_ft, 
                          stimp=10, slope_percent=0, handedness='right'):
        params = self.calculate_swing_parameters(
            core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent)
        
        # Generate tones with consistent lengths
        backswing_tone = self.generate_tone(
            params['backswing_time'], 400, 600)
        
        downswing_tone = self.generate_tone(
            params['dsi_time'], 600, 400)
        
        impact_chirp = self.generate_impact_chirp()
        backswing_beep = self.generate_tone(
            self.backswing_beep_duration, 1200, 1200)
        
        # Create stereo panning
        if handedness == 'right':
            pan_backswing = np.linspace(1, 0, len(backswing_tone))
            pan_downswing = np.linspace(0, 1, len(downswing_tone))
        else:
            pan_backswing = np.linspace(0, 1, len(backswing_tone))
            pan_downswing = np.linspace(1, 0, len(downswing_tone))

        # Combine channels with length checking
        min_length = min(len(backswing_tone), len(downswing_tone))
        left_channel = np.concatenate([
            backswing_tone[:min_length] * pan_backswing[:min_length],
            downswing_tone[:min_length] * pan_downswing[:min_length]
        ])
        right_channel = np.concatenate([
            backswing_tone[:min_length] * (1 - pan_backswing[:min_length]),
            downswing_tone[:min_length] * (1 - pan_downswing[:min_length])
        ])

        # Add audio cues with length checking
        beep_pos = min(int(0.9 * len(backswing_tone)), len(left_channel) - len(backswing_beep))
        left_channel[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep[:len(left_channel)-beep_pos] * 0.5
        right_channel[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep[:len(right_channel)-beep_pos] * 0.5

        impact_pos = min(len(backswing_tone), len(left_channel) - len(impact_chirp))
        left_channel[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp[:len(left_channel)-impact_pos] * 0.7
        right_channel[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp[:len(right_channel)-impact_pos] * 0.7

        # Normalize
        max_amplitude = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_amplitude > 0:
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
            st.error(f"Audio export failed: {str(e)}")
            st.info("Make sure FFmpeg is installed. On Streamlit Cloud, add 'ffmpeg' to packages.txt")
            
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("""
Based on SoundTempo Putt™ technology - Golf training system for developing consistent putting rhythm.  
*Note: For best results, use headphones to hear the stereo panning effects.*
""")