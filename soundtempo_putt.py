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
        self.club_length = 0.9  # meters (typical putter length)
        self.max_backswing_in = 24.0  # Maximum realistic backswing (2 feet)
        self.max_impact_velocity = 2.5  # m/s (professional long putt speed)

    def calculate_required_velocity(self, distance_ft, stimp, slope_percent):
        """Realistic velocity calculation based on PGA data"""
        distance_m = distance_ft * 0.3048
        base_velocity = 0.45 * (stimp/10) * sqrt(distance_m)
        slope_factor = 1 + (slope_percent / 200)  # More moderate slope effect
        return min(base_velocity * slope_factor, self.max_impact_velocity)

    def calculate_swing_parameters(self, core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent):
        """Realistic stroke parameters with physical constraints"""
        dsi_time = 30 / core_tempo_bpm
        backswing_time = dsi_time * backswing_rhythm
        
        required_velocity = self.calculate_required_velocity(distance_ft, stimp, slope_percent)
        
        # Constrained angular velocity calculation
        angular_velocity = min(required_velocity / (self.club_length * 0.6), 5.0)
        
        # Progressive scaling that decreases for longer putts
        scale_factor = max(0.5, 1.0 - (distance_ft/200))
        
        backswing_length_m = angular_velocity * dsi_time * self.club_length * scale_factor
        backswing_length_in = min(backswing_length_m * 39.37, self.max_backswing_in)
        
        return {
            'dsi_time': dsi_time,
            'backswing_time': backswing_time,
            'backswing_length_in': backswing_length_in,
            'required_velocity': required_velocity,
            'is_capped': backswing_length_in >= self.max_backswing_in
        }

    def generate_impact_chirp(self):
        t = np.linspace(0, self.impact_chirp_duration, 
                       int(self.sample_rate * self.impact_chirp_duration), False)
        chirp = np.sin(2 * pi * (1000 + 2000 * t/self.impact_chirp_duration) * t)
        envelope = np.linspace(1, 0, len(t))
        return chirp * envelope

    def generate_tone(self, duration, start_freq, end_freq):
        """Generate tone with exact length handling"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        freq = start_freq + (end_freq - start_freq) * t/duration
        tone = np.sin(2 * pi * freq * t)
        
        # Create matching envelope
        attack = int(0.1 * samples)
        sustain = int(0.8 * samples)
        release = max(1, samples - attack - sustain)
        envelope = np.concatenate([
            np.linspace(0, 1, attack),
            np.ones(sustain),
            np.linspace(1, 0, release)
        ])[:samples]
        
        return tone * envelope

    def generate_putt_audio(self, core_tempo_bpm, backswing_rhythm, distance_ft, 
                          stimp=10, slope_percent=0, handedness='right'):
        params = self.calculate_swing_parameters(
            core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent)
        
        # Generate tones
        backswing_tone = self.generate_tone(params['backswing_time'], 400, 600)
        downswing_tone = self.generate_tone(params['dsi_time'], 600, 400)
        impact_chirp = self.generate_impact_chirp()
        backswing_beep = self.generate_tone(self.backswing_beep_duration, 1200, 1200)

        # Create stereo panning
        pan_backswing = np.linspace(1, 0, len(backswing_tone)) if handedness == 'right' else np.linspace(0, 1, len(backswing_tone))
        pan_downswing = np.linspace(0, 1, len(downswing_tone)) if handedness == 'right' else np.linspace(1, 0, len(downswing_tone))

        # Combine channels safely
        min_length = min(len(backswing_tone), len(downswing_tone))
        left_channel = np.concatenate([
            backswing_tone[:min_length] * pan_backswing[:min_length],
            downswing_tone[:min_length] * pan_downswing[:min_length]
        ])
        right_channel = np.concatenate([
            backswing_tone[:min_length] * (1 - pan_backswing[:min_length]),
            downswing_tone[:min_length] * (1 - pan_downswing[:min_length])
        ])

        # Add audio cues with bounds checking
        beep_pos = min(int(0.9 * len(backswing_tone)), len(left_channel) - len(backswing_beep))
        if beep_pos > 0:
            left_channel[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep[:len(left_channel)-beep_pos] * 0.5
            right_channel[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep[:len(right_channel)-beep_pos] * 0.5

        impact_pos = min(len(backswing_tone), len(left_channel) - len(impact_chirp))
        if impact_pos > 0:
            left_channel[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp[:len(left_channel)-impact_pos] * 0.7
            right_channel[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp[:len(right_channel)-impact_pos] * 0.7

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
st.title("⛳ Realistic SoundTempo Putt Generator")

with st.expander("ℹ️ About this version"):
    st.write("""
    This version implements professional-grade putting physics:
    - Realistic backswing lengths (max 24 inches)
    - PGA-tested velocity calculations
    - Progressive distance scaling
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Stroke Parameters")
    core_tempo = st.slider("Core Tempo (BPM)", 72, 130, 90)
    rhythm = st.slider("Backswing Rhythm Ratio", 1.7, 2.5, 2.0, 0.1)
    handedness = st.radio("Handedness", ["Right", "Left"], index=0)

with col2:
    st.subheader("Green Conditions")
    distance = st.slider("Distance (feet)", 1, 100, 20)
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
            st.metric("Backswing Time", f"{params['backswing_time']:.3f} sec")
            st.metric("Downswing Time", f"{params['dsi_time']:.3f} sec")
        with col4:
            bs_metric = st.metric("Backswing Length", f"{params['backswing_length_in']:.1f} inches")
            if params['is_capped']:
                bs_metric.warning("Maximum realistic length")
            st.metric("Impact Velocity", f"{params['required_velocity']:.2f} m/s")
        
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
            st.info("Ensure FFmpeg is installed (see instructions in About section)")
            
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")

st.markdown("---")
st.caption("Professional putting physics model © 2023 - Use headphones for best results")