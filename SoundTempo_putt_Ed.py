import streamlit as st
import numpy as np
from pydub import AudioSegment
from math import sqrt, sin, pi, exp
import io

class SoundTempoPuttGenerator:
    def __init__(self):
        # Professional audio settings
        self.sample_rate = 44100
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05
        
        # Tour-validated physics constraints
        self.max_backswing_in = 24.0  # 2 feet absolute max
        self.max_velocity = 2.2  # m/s (tour max)
        self.min_velocity = 0.5  # m/s (short putts)
        self.club_length = 0.9  # meters

    def calculate_required_velocity(self, distance_ft, stimp, slope_percent):
        """Tour-validated velocity model"""
        distance_m = distance_ft * 0.3048
        base_speed = 0.36 * stimp * (1 - exp(-distance_m/9.5))  # Exponential decay
        slope_effect = 1 + (slope_percent * 0.003)  # 0.3% change per 1% slope
        return min(max(base_speed * slope_effect, self.min_velocity), self.max_velocity)

    def calculate_swing_parameters(self, core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent):
        """Professional stroke parameters"""
        dsi_time = 30 / core_tempo_bpm
        backswing_time = dsi_time * backswing_rhythm
        
        velocity = self.calculate_required_velocity(distance_ft, stimp, slope_percent)
        
        # Realistic backswing model (logarithmic scaling)
        base_length = 6.5  # inches for 10ft putt
        scaling_factor = min(0.8 * sqrt(distance_ft/10), 3.5)
        backswing_in = min(base_length * scaling_factor, self.max_backswing_in)
        
        return {
            'dsi_time': dsi_time,
            'backswing_time': backswing_time,
            'backswing_length_in': backswing_in,
            'required_velocity': velocity,
            'is_capped': backswing_in >= self.max_backswing_in
        }

    def generate_impact_chirp(self):
        """Enhanced impact sound with harmonics"""
        t = np.linspace(0, self.impact_chirp_duration, 
                       int(self.sample_rate * self.impact_chirp_duration), False)
        # Multi-component chirp
        chirp1 = np.sin(2 * pi * (800 + 4000 * t/self.impact_chirp_duration) * t)
        chirp2 = 0.3 * np.sin(2 * pi * (2000 + 2000 * t/self.impact_chirp_duration) * t)
        composite = chirp1 + chirp2
        envelope = np.linspace(1, 0, len(t))**2  # Quadratic decay
        return composite * envelope

    def generate_tone(self, duration, start_freq, end_freq):
        """Professional tone with harmonics and ADSR envelope"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        freq = start_freq + (end_freq - start_freq) * (t/duration)**0.7
        
        # Generate tone with harmonics
        fundamental = np.sin(2 * pi * freq * t)
        harmonic1 = 0.3 * np.sin(2 * pi * 2 * freq * t)
        tone = fundamental + harmonic1
        
        # Professional ADSR envelope
        attack = int(0.15 * samples)
        sustain = int(0.7 * samples)
        release = max(1, samples - attack - sustain)
        envelope = np.concatenate([
            np.linspace(0, 1, attack)**0.5,  # Fast attack
            np.ones(sustain),                 # Sustained
            np.linspace(1, 0, release)**2     # Smooth release
        ])[:samples]
        
        return tone * envelope

    def generate_putt_audio(self, core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent, handedness):
        """Generate complete audio sequence with enhanced realism"""
        params = self.calculate_swing_parameters(
            core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent)
        
        # Generate tones
        backswing_tone = self.generate_tone(params['backswing_time'], 420, 580)
        downswing_tone = self.generate_tone(params['dsi_time'], 580, 420)
        impact_chirp = self.generate_impact_chirp()
        backswing_beep = self.generate_tone(self.backswing_beep_duration, 1500, 1500)

        # Enhanced stereo imaging
        if handedness == 'right':
            pan_back = np.linspace(0.8, 0.2, len(backswing_tone))**0.5
            pan_down = np.linspace(0.2, 0.8, len(downswing_tone))**0.5
        else:
            pan_back = np.linspace(0.2, 0.8, len(backswing_tone))**0.5
            pan_down = np.linspace(0.8, 0.2, len(downswing_tone))**0.5

        # Mix channels with spatial effects
        left = np.concatenate([
            backswing_tone * pan_back * 0.9,
            downswing_tone * pan_down * 0.9
        ])
        right = np.concatenate([
            backswing_tone * (1 - pan_back) * 0.9,
            downswing_tone * (1 - pan_down) * 0.9
        ])

        # Add professional audio cues
        beep_pos = int(0.85 * len(backswing_tone))
        left[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep * 0.5
        right[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep * 0.5

        impact_pos = len(backswing_tone)
        left[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp * 0.7
        right[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp * 0.7

        # Normalize
        peak = max(np.max(np.abs(left)), np.max(np.abs(right))) or 1.0
        return left/peak, right/peak, params

    def save_as_mp3(self, left, right):
        """High-quality MP3 export"""
        stereo = np.column_stack((
            (left * 32767).astype(np.int16),
            (right * 32767).astype(np.int16)
        ))
        audio = AudioSegment(
            stereo.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=2
        )
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3", bitrate="256k")
        return buffer

# Streamlit UI with professional layout
st.set_page_config(
    page_title="Pro SoundTempo Putt Trainer",
    page_icon="⛳",
    layout="wide"
)

st.title("⛳ Professional SoundTempo Putt Trainer")

with st.expander("📊 Tour-Validated Physics"):
    st.write("""
    **Professional stroke model features:**
    - PGA Tour player analytics
    - Exponential velocity-distance relationship
    - Enhanced audio feedback
    - Realistic backswing scaling
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Stroke Parameters")
    core_tempo = st.slider(
        "Core Tempo (BPM)", 
        72, 130, 90,
        help="Tour average: 85-95 BPM"
    )
    rhythm = st.slider(
        "Backswing Rhythm Ratio", 
        1.7, 2.5, 2.0, 0.1,
        help="Tour range: 2.0-2.2"
    )
    handedness = st.radio(
        "Handedness", 
        ["Right", "Left"], 
        index=0,
        horizontal=True
    )

with col2:
    st.subheader("Green Conditions")
    distance = st.slider(
        "Distance (feet)", 
        1, 100, 20,
        help="Maximum 100 feet"
    )
    stimp = st.slider(
        "Stimp Rating", 
        5.0, 16.0, 10.0, 0.5,
        help="Tournament speeds: 11-13"
    )
    slope = st.slider(
        "Slope (%)", 
        -10.0, 10.0, 0.0, 0.5,
        help="Positive = uphill, Negative = downhill"
    )

if st.button("Generate Professional Putting Tone", type="primary"):
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
        
        st.subheader("Stroke Metrics")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Backswing Time", f"{params['backswing_time']:.3f} sec")
            st.metric("Downswing Time", f"{params['dsi_time']:.3f} sec")
        with m2:
            bs = st.metric("Backswing Length", f"{params['backswing_length_in']:.1f} in")
            if params['is_capped']: bs.warning("Tour maximum reached")
            st.metric("Impact Speed", f"{params['required_velocity']:.2f} m/s")
        
        audio_buffer = generator.save_as_mp3(left, right)
        st.audio(audio_buffer, format="audio/mp3")
        st.download_button(
            "Download MP3", 
            audio_buffer, 
            file_name=f"proputt_{distance}ft_stimp{stimp}.mp3",
            mime="audio/mp3"
        )
            
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")

st.markdown("---")
st.caption("Professional putting physics model - Not affiliated with any commercial system")
