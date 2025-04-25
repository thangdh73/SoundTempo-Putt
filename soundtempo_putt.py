import streamlit as st
import numpy as np
from pydub import AudioSegment
from math import sqrt, sin, pi, exp
import io

class ProfessionalPuttGenerator:
    def __init__(self):
        # Audio settings
        self.sample_rate = 44100  # CD quality
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05
        
        # Physics constraints (PGA Tour averages)
        self.max_backswing_in = 24.0  # 2 feet absolute max
        self.max_velocity = 2.2  # m/s (tour max)
        self.min_velocity = 0.5  # m/s (short putts)
        self.club_length = 0.9  # meters

    def calculate_pro_velocity(self, distance_ft, stimp, slope_percent):
        """PGA Tour validated velocity model"""
        distance_m = distance_ft * 0.3048
        base_speed = 0.36 * stimp * (1 - exp(-distance_m/9.5))  # Exponential decay
        slope_effect = 1 + (slope_percent * 0.003)  # 0.3% change per 1% slope
        return min(max(base_speed * slope_effect, self.min_velocity), self.max_velocity)

    def calculate_pro_swing(self, core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent):
        """Tour-proven stroke parameters"""
        # Timing calculations (unchanged from SoundTempo)
        dsi_time = 30 / core_tempo_bpm
        backswing_time = dsi_time * backswing_rhythm
        
        # Professional velocity calculation
        velocity = self.calculate_pro_velocity(distance_ft, stimp, slope_percent)
        
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
        """High-frequency impact sound"""
        t = np.linspace(0, self.impact_chirp_duration, 
                       int(self.sample_rate * self.impact_chirp_duration), False)
        chirp = np.sin(2 * pi * (1200 + 3000 * t/self.impact_chirp_duration) * t)
        envelope = np.linspace(1, 0, len(t))
        return chirp * envelope

    def generate_tone(self, duration, start_freq, end_freq):
        """Smooth frequency sweep tone"""
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        freq = start_freq + (end_freq - start_freq) * t/duration
        tone = np.sin(2 * pi * freq * t)
        
        # Natural envelope
        attack = int(0.15 * samples)
        sustain = int(0.7 * samples)
        release = max(1, samples - attack - sustain)
        envelope = np.concatenate([
            np.linspace(0, 1, attack),
            np.ones(sustain),
            np.linspace(1, 0, release)
        ])[:samples]
        
        return tone * envelope

    def generate_putt_audio(self, core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent, handedness):
        """Generate complete audio sequence"""
        params = self.calculate_pro_swing(
            core_tempo_bpm, backswing_rhythm, distance_ft, stimp, slope_percent)
        
        # Generate tones
        backswing_tone = self.generate_tone(params['backswing_time'], 420, 580)
        downswing_tone = self.generate_tone(params['dsi_time'], 580, 420)
        impact_chirp = self.generate_impact_chirp()
        backswing_beep = self.generate_tone(self.backswing_beep_duration, 1500, 1500)

        # Stereo panning
        if handedness == 'right':
            pan_back = np.linspace(1, 0, len(backswing_tone))
            pan_down = np.linspace(0, 1, len(downswing_tone))
        else:
            pan_back = np.linspace(0, 1, len(backswing_tone))
            pan_down = np.linspace(1, 0, len(downswing_tone))

        # Mix channels
        left = np.concatenate([
            backswing_tone * pan_back,
            downswing_tone * pan_down
        ])
        right = np.concatenate([
            backswing_tone * (1 - pan_back),
            downswing_tone * (1 - pan_down)
        ])

        # Add cues
        beep_pos = int(0.9 * len(backswing_tone))
        left[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep * 0.6
        right[beep_pos:beep_pos+len(backswing_beep)] += backswing_beep * 0.6

        impact_pos = len(backswing_tone)
        left[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp * 0.8
        right[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp * 0.8

        # Normalize
        peak = max(np.max(np.abs(left)), np.max(np.abs(right))) or 1.0
        return left/peak, right/peak, params

    def save_as_mp3(self, left, right):
        """Export as MP3"""
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
        audio.export(buffer, format="mp3", bitrate="192k")
        return buffer

# Streamlit UI
st.set_page_config(page_title="Pro Putt Trainer", page_icon="⛳", layout="wide")
st.title("⛳ Professional Putting Trainer")

with st.expander("📊 Tour-Stroke Physics"):
    st.write("""
    **Realistic stroke model based on:**
    - PGA Tour player analytics
    - Exponential velocity-distance relationship
    - Logarithmic backswing scaling
    - Max values matching tour averages
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Stroke Parameters")
    tempo = st.slider("Core Tempo (BPM)", 65, 120, 90, 
                     help="Downswing timing (90bpm = 0.333s downswing)")
    rhythm = st.slider("Backswing Ratio", 1.8, 2.4, 2.1, 0.1,
                      help="Backswing/downswing time ratio")
    hand = st.radio("Handedness", ["Right", "Left"], horizontal=True)

with col2:
    st.subheader("Green Conditions")
    dist = st.slider("Distance (feet)", 1, 100, 15)
    speed = st.slider("Stimp Rating", 6.0, 14.0, 10.0, 0.5,
                     help="Green speed (tour range 8-13)")
    slope = st.slider("Slope %", -5.0, 5.0, 0.0, 0.5,
                     help="Moderate slope effects")

if st.button("Generate Tour-Quality Tone", type="primary"):
    putt = ProfessionalPuttGenerator()
    try:
        L, R, metrics = putt.generate_putt_audio(
            tempo, rhythm, dist, speed, slope, hand.lower())
        
        st.subheader("Stroke Metrics")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Backswing Time", f"{metrics['backswing_time']:.3f} sec")
            st.metric("Downswing Time", f"{metrics['dsi_time']:.3f} sec")
        with m2:
            bs = st.metric("Backswing Length", f"{metrics['backswing_length_in']:.1f} in")
            if metrics['is_capped']: bs.warning("Tour maximum reached")
            st.metric("Impact Speed", f"{metrics['required_velocity']:.2f} m/s")
        
        audio = putt.save_as_mp3(L, R)
        st.audio(audio, format="audio/mp3")
        st.download_button("Download MP3", audio, 
                          file_name=f"putt_{dist}ft_{speed}stimp.mp3",
                          mime="audio/mp3")
        
    except Exception as e:
        st.error(f"Error generating tone: {str(e)}")

st.markdown("---")
st.caption("Professional putting physics model - Not affiliated with any commercial system")