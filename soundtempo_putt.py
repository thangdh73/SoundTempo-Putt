import streamlit as st
import numpy as np
from pydub import AudioSegment
from math import sqrt, sin, pi
import io

class SoundTempoPuttGenerator:
    def __init__(self):
        self.sample_rate = 44100  # CD-quality audio
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05
        self.stimp_conversion = 0.3048 / 1.83
        self.club_length = 0.9
        self.gravity = 9.81

    def safe_audio_operation(self, func, *args, **kwargs):
        """Wrapper to safely handle audio operations"""
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if "broadcast" in str(e):
                st.warning("Adjusting audio lengths to fix size mismatch...")
                # Find the shortest length among arrays
                min_len = min(len(arg) for arg in args if hasattr(arg, '__len__'))
                # Truncate all arrays to the shortest length
                truncated_args = [arg[:min_len] if hasattr(arg, '__len__') else arg for arg in args]
                return func(*truncated_args, **kwargs)
            raise

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
        return self.safe_audio_operation(lambda: chirp * envelope)

    def generate_tone(self, duration, start_freq, end_freq):
        """Generate tone with exact length handling"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        freq = start_freq + (end_freq - start_freq) * t/duration
        tone = np.sin(2 * pi * freq * t)
        
        # Create matching envelope
        attack = int(0.1 * samples)
        sustain = int(0.8 * samples)
        release = max(1, samples - attack - sustain)  # Ensure at least 1 sample
        envelope = np.concatenate([
            np.linspace(0, 1, attack),
            np.ones(sustain),
            np.linspace(1, 0, release)
        ])[:samples]  # Ensure exact length
        
        return self.safe_audio_operation(lambda: tone * envelope)

    def generate_putt_audio(self, core_tempo_bpm, backswing_rhythm, distance_ft, 
                          stimp=10, slope_percent=0, handedness='right'):
        params = self.calculate_swing_parameters(
            core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent)
        
        # Generate tones with safe operations
        backswing_tone = self.generate_tone(params['backswing_time'], 400, 600)
        downswing_tone = self.generate_tone(params['dsi_time'], 600, 400)
        impact_chirp = self.generate_impact_chirp()
        backswing_beep = self.generate_tone(self.backswing_beep_duration, 1200, 1200)

        # Create stereo panning with safe operations
        pan_backswing = np.linspace(1, 0, len(backswing_tone)) if handedness == 'right' else np.linspace(0, 1, len(backswing_tone))
        pan_downswing = np.linspace(0, 1, len(downswing_tone)) if handedness == 'right' else np.linspace(1, 0, len(downswing_tone))

        # Combine channels safely
        left_channel = self.safe_audio_operation(
            np.concatenate,
            [
                backswing_tone * pan_backswing,
                downswing_tone * pan_downswing
            ]
        )
        right_channel = self.safe_audio_operation(
            np.concatenate,
            [
                backswing_tone * (1 - pan_backswing),
                downswing_tone * (1 - pan_downswing)
            ]
        )

        # Add audio cues with bounds checking
        def safe_add(target, source, position):
            if position < 0 or position >= len(target):
                return target
            available = len(target) - position
            if available <= 0:
                return target
            target[position:position+min(len(source), available)] += source[:available]
            return target

        beep_pos = int(0.9 * len(backswing_tone))
        left_channel = safe_add(left_channel, backswing_beep * 0.5, beep_pos)
        right_channel = safe_add(right_channel, backswing_beep * 0.5, beep_pos)

        impact_pos = len(backswing_tone)
        left_channel = safe_add(left_channel, impact_chirp * 0.7, impact_pos)
        right_channel = safe_add(right_channel, impact_chirp * 0.7, impact_pos)

        # Normalize
        max_amplitude = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel))) or 1.0
        return left_channel/max_amplitude, right_channel/max_amplitude, params

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
st.markdown("Create customized putting tones based on green conditions and stroke mechanics.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Stroke Parameters")
    core_tempo = st.slider("Core Tempo (BPM)", 72, 130, 90)
    rhythm = st.slider("Backswing Rhythm Ratio", 1.7, 2.5, 2.0, 0.1)
    handedness = st.radio("Handedness", ["Right", "Left"], index=0)

with col2:
    st.subheader("Green Conditions")
    distance = st.slider("Distance (feet)", 1, 60, 15)
    stimp = st.slider("Stimp Rating", 5.0, 16.0, 10.0, 0.5)
    slope = st.slider("Slope (%)", -10.0, 10.0, 0.0, 0.5)

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
            st.info("""
            If you're seeing FFmpeg errors:
            1. For local runs: Install FFmpeg (sudo apt install ffmpeg / brew install ffmpeg)
            2. For Streamlit Cloud: Ensure 'ffmpeg' is in packages.txt
            """)
            
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        if "broadcast" in str(e):
            st.info("Try adjusting the tempo or rhythm slightly. The system will automatically fix size mismatches.")

st.markdown("---")
st.caption("Based on SoundTempo Putt™ technology - Use headphones for best stereo effects.")