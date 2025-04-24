import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from math import sqrt

class SoundTempoPutt:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 44100
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05
        
        # Physics parameters
        self.typical_impact_velocity = 1.2  # m/s for 10ft putt on Stimp 10
        self.putter_head_weight = 0.3  # kg
        self.club_length = 0.9  # meters

    # Physics Calculations -----------------------------------------------------
    def calculate_swing_parameters(self, core_tempo_bpm, backswing_rhythm, 
                                 distance_ft, stimp, slope_percent=0):
        """Calculate timing and backswing length based on distance"""
        # 1. Calculate fixed timing (Law of Isochrony)
        dsi_time = 30 / core_tempo_bpm  # Downswing-to-impact time
        backswing_time = dsi_time * backswing_rhythm
        
        # 2. Calculate required ball velocity
        distance_m = distance_ft * 0.3048
        stimp_mps = stimp * 0.3048 / 1.83  # Convert stimp to m/s
        slope_factor = 1 + (slope_percent / 100)
        required_velocity = stimp_mps * sqrt(distance_m * slope_factor * 2)
        
        # 3. Calculate backswing length (empirical relationship)
        angular_velocity = required_velocity / (self.club_length * 0.7)
        backswing_length_m = angular_velocity * dsi_time * self.club_length * 1.2
        backswing_length_in = backswing_length_m * 39.37
        
        return {
            'dsi_time': dsi_time,
            'backswing_time': backswing_time,
            'total_time': backswing_time + dsi_time,
            'backswing_length': backswing_length_in,
            'required_velocity': required_velocity
        }

    # Audio Generation --------------------------------------------------------
    def generate_impact_chirp(self):
        """Generate the distinct impact chirp sound"""
        t = np.linspace(0, self.impact_chirp_duration, 
                       int(self.sample_rate * self.impact_chirp_duration), False)
        chirp = np.sin(2 * np.pi * (1000 + 2000 * t/self.impact_chirp_duration) * t)
        envelope = np.concatenate([
            np.linspace(0, 1, int(0.01 * self.sample_rate)),
            np.linspace(1, 0, int(0.04 * self.sample_rate))
        ])
        return chirp * envelope
    
    def generate_backswing_beep(self):
        """Generate the backswing beep cue"""
        t = np.linspace(0, self.backswing_beep_duration, 
                       int(self.sample_rate * self.backswing_beep_duration), False)
        beep = np.sin(2 * np.pi * 1200 * t)
        envelope = np.concatenate([
            np.linspace(0, 1, int(0.01 * self.sample_rate)),
            np.linspace(1, 0, int(0.04 * self.sample_rate))
        ])
        return beep * envelope
    
    def generate_putt_audio(self, core_tempo, backswing_rhythm, distance_ft, 
                          stimp=10, slope=0, handedness='right'):
        """Generate complete putt audio with distance calculations"""
        # Calculate timing and length
        params = self.calculate_swing_parameters(
            core_tempo, backswing_rhythm, distance_ft, stimp, slope)
        
        print(f"Generating {distance_ft}ft putt:")
        print(f"- Backswing: {params['backswing_time']:.3f}s ({params['backswing_length']:.1f}in)")
        print(f"- Downswing: {params['dsi_time']:.3f}s")
        print(f"- Total time: {params['total_time']:.3f}s")
        
        # Generate time arrays
        t_back = np.linspace(0, params['backswing_time'], 
                            int(self.sample_rate * params['backswing_time']), False)
        t_down = np.linspace(0, params['dsi_time'], 
                            int(self.sample_rate * params['dsi_time']), False)
        
        # Generate tones with distance-appropriate pitch modulation
        back_tone = np.sin(2 * np.pi * (400 + 300 * t_back/params['backswing_time']) * t_back)
        down_tone = np.sin(2 * np.pi * (700 - 300 * t_down/params['dsi_time']) * t_down)
        
        # Apply envelopes
        back_env = np.concatenate([
            np.linspace(0, 1, int(0.1 * len(t_back))),
            np.ones(int(0.8 * len(t_back))),
            np.linspace(1, 0, int(0.1 * len(t_back)))
        ])
        down_env = np.concatenate([
            np.linspace(0, 1, int(0.1 * len(t_down))),
            np.ones(int(0.7 * len(t_down))),
            np.linspace(1, 0, int(0.2 * len(t_down)))
        ])
        
        back_tone *= back_env
        down_tone *= down_env
        
        # Add audio cues
        beep_pos = int(0.9 * len(back_tone))
        impact_pos = len(back_tone)
        chirp = self.generate_impact_chirp()
        beep = self.generate_backswing_beep()
        
        # Create stereo panning
        if handedness == 'right':
            pan_back = np.linspace(1, 0, len(back_tone))  # Left to right
            pan_down = np.linspace(0, 1, len(down_tone))  # Right to left
        else:
            pan_back = np.linspace(0, 1, len(back_tone))  # Right to left
            pan_down = np.linspace(1, 0, len(down_tone))  # Left to right
            
        # Apply panning
        left_channel = np.concatenate([
            back_tone * pan_back,
            down_tone * pan_down
        ])
        right_channel = np.concatenate([
            back_tone * (1 - pan_back),
            down_tone * (1 - pan_down)
        ])
        
        # Add beeps
        left_channel[beep_pos:beep_pos+len(beep)] += beep * 0.7
        right_channel[beep_pos:beep_pos+len(beep)] += beep * 0.7
        left_channel[impact_pos:impact_pos+len(chirp)] += chirp * 0.7
        right_channel[impact_pos:impact_pos+len(chirp)] += chirp * 0.7
        
        # Normalize
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        left_channel /= max_val
        right_channel /= max_val
        
        return left_channel, right_channel, params
    
    def generate_rhythm_audio(self, core_tempo, cycles=10, handedness='right'):
        """Generate rhythm mode sequence"""
        # Fixed 2.0 rhythm in rhythm mode
        params = self.calculate_swing_parameters(core_tempo, 2.0, 10, 10)
        
        # Generate one putt cycle
        left, right, _ = self.generate_putt_audio(core_tempo, 2.0, 10, 10, 0, handedness)
        
        # Reverse for return swing
        left_rev = left[::-1]
        right_rev = right[::-1]
        
        # Create full cycle (back and forth)
        cycle_left = np.concatenate([left, right_rev])
        cycle_right = np.concatenate([right, left_rev])
        
        # Repeat for specified cycles
        full_left = np.tile(cycle_left, cycles)
        full_right = np.tile(cycle_right, cycles)
        
        return full_left, full_right, params
    
    # File Handling -----------------------------------------------------------
    def save_as_mp3(self, left_channel, right_channel, filename):
        """Save stereo audio as MP3"""
        stereo = np.column_stack((
            (left_channel * 32767).astype(np.int16),
            (right_channel * 32767).astype(np.int16)
        ))
        
        audio = AudioSegment(
            stereo.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=2
        )
        audio.export(filename, format="mp3", bitrate="192k")
        print(f"Saved: {filename}")

# Example Usage --------------------------------------------------------------
if __name__ == "__main__":
    stp = SoundTempoPutt()
    
    # 1. Generate different distance putts (same timing, different implied lengths)
    for distance in [5, 15, 30]:  # Short, medium, long putts
        left, right, params = stp.generate_putt_audio(
            core_tempo=90,
            backswing_rhythm=2.0,
            distance_ft=distance,
            stimp=10,
            handedness='right'
        )
        stp.save_as_mp3(left, right, f"putt_{distance}ft.mp3")
    
    # 2. Generate rhythm mode sequence
    left_rhythm, right_rhythm, _ = stp.generate_rhythm_audio(
        core_tempo=90,
        cycles=10,
        handedness='right'
    )
    stp.save_as_mp3(left_rhythm, right_rhythm, "rhythm_mode_90bpm.mp3")
    
    # 3. Visualize distance vs backswing length
    distances = np.linspace(1, 60, 20)
    lengths = [stp.calculate_swing_parameters(90, 2.0, d, 10)['backswing_length'] 
               for d in distances]
    
    plt.figure(figsize=(10,5))
    plt.plot(distances, lengths, 'b-o')
    plt.title("Backswing Length vs. Putt Distance (90bpm, Rhythm 2.0, Stimp 10)")
    plt.xlabel("Distance (feet)")
    plt.ylabel("Backswing Length (inches)")
    plt.grid(True)
    plt.savefig("distance_vs_length.png")
    plt.show()