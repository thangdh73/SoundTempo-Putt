import streamlit as st
import numpy as np
from pydub import AudioSegment
from math import sqrt, sin, pi
import io

class SoundTempoPuttGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05
        self.stimp_conversion = 0.3048 / 1.83  # Empirical conversion factor
        self.club_length = 0.9  # meters (typical putter length)
        self.gravity = 9.81
        self.max_backswing_in = 24.0  # Maximum realistic backswing (2 feet)

    def calculate_required_velocity(self, distance_ft, stimp, slope_percent):
        """Calculate required ball velocity with more realistic constraints"""
        distance_m = distance_ft * 0.3048
        stimp_mps = stimp * self.stimp_conversion
        slope_factor = 1 + (slope_percent / 100)
        
        # Adjusted formula with more realistic coefficients
        required_velocity = sqrt(2 * self.gravity * 0.25 * stimp_mps * distance_m) * slope_factor
        return min(required_velocity, 3.0)  # Cap at 3 m/s (unlikely to exceed in putting)

    def calculate_swing_parameters(self, core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent):
        """Calculate parameters with realistic constraints"""
        dsi_time = 30 / core_tempo_bpm
        backswing_time = dsi_time * backswing_rhythm
        
        required_velocity = self.calculate_required_velocity(distance_ft, stimp, slope_percent)
        
        # More realistic angular velocity calculation
        angular_velocity = required_velocity / (self.club_length * 0.65)  # Adjusted lever arm
        
        # Revised backswing length calculation with empirical scaling
        backswing_length_m = angular_velocity * dsi_time * self.club_length * 0.8  # Reduced empirical factor
        
        # Convert to inches and apply maximum constraint
        backswing_length_in = min(backswing_length_m * 39.37, self.max_backswing_in)
        
        return {
            'dsi_time': dsi_time,
            'backswing_time': backswing_time,
            'backswing_length_in': backswing_length_in,
            'required_velocity': required_velocity,
            'is_capped': backswing_length_in >= self.max_backswing_in
        }

    # ... (rest of the audio generation methods remain the same) ...

# Streamlit UI
st.title("⛳ Realistic SoundTempo Putt Generator")

with st.expander("ℹ️ About this version"):
    st.write("""
    This version implements more realistic putting physics:
    - Capped maximum backswing length (24 inches)
    - Adjusted velocity calculations
    - More accurate angular momentum modeling
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
                bs_metric.warning("Maximum realistic length reached")
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
            
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")

st.markdown("---")
st.caption("Based on SoundTempo Putt™ technology with realistic physics adjustments")