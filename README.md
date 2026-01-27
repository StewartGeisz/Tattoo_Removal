# Surgical Video Anonymizer

## Overview
Advanced surgical video anonymizer that uses temporal tracking to blur faces and tattoos while maintaining smooth, consistent coverage across video frames.

## Core Implementation
- **`smooth_tattoo_tracker.py`** - Main anonymizer with temporal tracking, enhanced tattoo detection, and smooth blur application

## Features
- **Face Detection**: OpenCV-based face detection with Gaussian blur
- **Tattoo Detection**: Multi-method approach combining dark region detection and pattern analysis
- **Temporal Tracking**: Smooth tracking across frames reduces jumpy blur boxes
- **Enhanced Coverage**: Multiple detection thresholds for comprehensive tattoo coverage
- **Real-time Processing**: Optimized for surgical video processing

## Usage
```bash
python smooth_tattoo_tracker.py input_video.mp4 output_video.mp4 [--debug]
```

## Requirements
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Input/Output
- **Input**: Place videos in `input/` folder
- **Output**: Processed videos saved to `output/` folder

## Technical Details
The implementation uses:
- Multi-threshold dark region detection for tattoos
- Enhanced pattern detection using edge and texture analysis
- Temporal smoothing with IoU-based tracking
- Adaptive blur sizing based on region dimensions