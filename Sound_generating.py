import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import simpleaudio as sa

class SoundTempoPutt:
    def __init__(self):
        self.sample_rate = 44100
        self.impact_chirp_duration = 0.05  # seconds
        self.backswing_beep_duration = 0.05  # seconds
        
    def generate_impact_chirp(self):
        """Generate the distinct impact chirp sound"""
        t = np.linspace(0, self.impact_chirp_duration, 
                       int(self.sample_rate * self.impact_chirp_duration), False)
        # Chirp with rising frequency
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
    
    def generate_putt_tone(self, core_tempo, backswing_rhythm, handedness='right'):
        """
        Generate a putting stroke tone with proper timing
        core_tempo: in BPM (e.g., 90)
        backswing_rhythm: ratio (e.g., 2.0)
        handedness: 'right' or 'left' for stereo panning
        """
        # Calculate timing in seconds
        dsi_time = 30 / core_tempo  # Downswing to impact time
        backswing_time = dsi_time * backswing_rhythm
        total_time = backswing_time + dsi_time
        
        # Generate time arrays
        t_backswing = np.linspace(0, backswing_time, int(self.sample_rate * backswing_time), False)
        t_downswing = np.linspace(0, dsi_time, int(self.sample_rate * dsi_time), False)
        
        # Generate tones with pitch change
        backswing_tone = np.sin(2 * np.pi * (400 + 200 * t_backswing/backswing_time) * t_backswing)
        downswing_tone = np.sin(2 * np.pi * (600 - 200 * t_downswing/dsi_time) * t_downswing)
        
        # Apply envelopes
        backswing_env = np.concatenate([
            np.linspace(0, 1, int(0.1 * backswing_time * self.sample_rate)),
            np.ones(int(0.8 * backswing_time * self.sample_rate)),
            np.linspace(1, 0, int(0.1 * backswing_time * self.sample_rate))
        ])
        
        downswing_env = np.concatenate([
            np.linspace(0, 1, int(0.1 * dsi_time * self.sample_rate)),
            np.ones(int(0.7 * dsi_time * self.sample_rate)),
            np.linspace(1, 0, int(0.2 * dsi_time * self.sample_rate))
        ])
        
        # Ensure envelope lengths match tone lengths
        if len(backswing_tone) != len(backswing_env):
            backswing_env = np.interp(np.linspace(0, 1, len(backswing_tone)), 
                                    np.linspace(0, 1, len(backswing_env)), 
                                    backswing_env)
        if len(downswing_tone) != len(downswing_env):
            downswing_env = np.interp(np.linspace(0, 1, len(downswing_tone)), 
                                    np.linspace(0, 1, len(downswing_env)), 
                                    downswing_env)
        
        backswing_tone = backswing_tone * backswing_env
        downswing_tone = downswing_tone * downswing_env
        
        # Combine with beeps
        backswing_beep_pos = int(0.9 * backswing_time * self.sample_rate)
        impact_chirp = self.generate_impact_chirp()
        backswing_beep = self.generate_backswing_beep()
        
        # Create stereo panning effect
        if handedness == 'right':
            pan_backswing = np.linspace(1, 0, len(backswing_tone))  # Left to right
            pan_downswing = np.linspace(0, 1, len(downswing_tone))  # Right to left
        else:
            pan_backswing = np.linspace(0, 1, len(backswing_tone))  # Right to left
            pan_downswing = np.linspace(1, 0, len(downswing_tone))  # Left to right
            
        # Apply panning
        backswing_tone_left = backswing_tone * pan_backswing
        backswing_tone_right = backswing_tone * (1 - pan_backswing)
        downswing_tone_left = downswing_tone * pan_downswing
        downswing_tone_right = downswing_tone * (1 - pan_downswing)
        
        # Combine all elements
        full_tone_left = np.concatenate([
            backswing_tone_left,
            downswing_tone_left
        ])
        
        full_tone_right = np.concatenate([
            backswing_tone_right,
            downswing_tone_right
        ])
        
        # Add beeps
        full_tone_left[backswing_beep_pos:backswing_beep_pos+len(backswing_beep)] += backswing_beep * 0.7
        full_tone_right[backswing_beep_pos:backswing_beep_pos+len(backswing_beep)] += backswing_beep * 0.7
        
        impact_pos = len(backswing_tone)
        full_tone_left[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp * 0.7
        full_tone_right[impact_pos:impact_pos+len(impact_chirp)] += impact_chirp * 0.7
        
        # Normalize
        max_val = max(np.max(np.abs(full_tone_left)), np.max(np.abs(full_tone_right)))
        full_tone_left = full_tone_left / max_val
        full_tone_right = full_tone_right / max_val
        
        return full_tone_left, full_tone_right, total_time
    
    def generate_rhythm_sequence(self, core_tempo, cycles=10, handedness='right'):
        """Generate rhythm mode sequence (10 cycles by default)"""
        # In rhythm mode, backswing rhythm is always 2.0
        dsi_time = 30 / core_tempo
        backswing_time = dsi_time * 2.0
        cycle_time = 2 * (backswing_time + dsi_time)  # Back and forth
        
        # Generate one cycle
        tone_left, tone_right, _ = self.generate_putt_tone(core_tempo, 2.0, handedness)
        
        # Reverse for the return swing
        tone_left_rev = tone_left[::-1]
        tone_right_rev = tone_right[::-1]
        
        # Combine for one full cycle (back and forth)
        cycle_left = np.concatenate([tone_left, tone_right_rev])
        cycle_right = np.concatenate([tone_right, tone_left_rev])
        
        # Repeat for specified cycles
        full_left = np.tile(cycle_left, cycles)
        full_right = np.tile(cycle_right, cycles)
        
        return full_left, full_right, cycle_time * cycles
    
    def save_as_mp3(self, left_channel, right_channel, filename):
        """Save stereo audio as MP3"""
        # Convert to 16-bit PCM
        audio_left = (left_channel * 32767).astype(np.int16)
        audio_right = (right_channel * 32767).astype(np.int16)
        
        # Combine channels
        stereo = np.column_stack((audio_left, audio_right))
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            stereo.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=2
        )
        
        # Export
        audio_segment.export(filename, format="mp3", bitrate="192k")
        print(f"Saved as {filename}")

# Example Usage
if __name__ == "__main__":
    stp = SoundTempoPutt()
    
    # Generate Putt Mode example (90bpm, 2.0 rhythm, right-handed)
    left, right, duration = stp.generate_putt_tone(90, 2.0, 'right')
    stp.save_as_mp3(left, right, "soundtempo_putt_mode.mp3")
    
    # Generate Rhythm Mode example (90bpm, 10 cycles, right-handed)
    left_rhythm, right_rhythm, duration = stp.generate_rhythm_sequence(90, 10, 'right')
    stp.save_as_mp3(left_rhythm, right_rhythm, "soundtempo_rhythm_mode.mp3")