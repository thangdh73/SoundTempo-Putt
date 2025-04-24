# SoundTempo Putt Generator

A Python application for generating audio feedback for golf putting practice. This tool helps golfers develop consistent putting rhythm by providing audio cues based on stroke mechanics and green conditions.

## Features

- Customizable putting stroke parameters (tempo, rhythm, handedness)
- Adjustable green conditions (distance, stimp rating, slope)
- Stereo audio panning for spatial feedback
- Real-time audio generation and playback
- Detailed stroke analysis metrics

## Requirements

- Python 3.8+
- FFmpeg (required for audio processing)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/soundtempo-putt.git
cd soundtempo-putt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

## Usage

Run the Streamlit app:
```bash
streamlit run soundtempo_putt.py
```

## Features

- **Stroke Parameters**
  - Core Tempo (BPM)
  - Backswing Rhythm Ratio
  - Handedness (Right/Left)

- **Green Conditions**
  - Distance (feet)
  - Stimp Rating
  - Slope (%)

- **Output**
  - Real-time audio feedback
  - Stroke analysis metrics
  - Downloadable MP3 files

## License

MIT License

## Acknowledgments

Based on SoundTempo Putt™ technology - Golf training system for developing consistent putting rhythm. 