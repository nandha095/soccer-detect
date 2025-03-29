# Football-stats

A computer vision-based football player tracking system that analyzes football videos to track and annotate player movements.

## Features

- Video processing and player tracking
- Object detection using YOLOv8 model
- Track visualization and annotation
- Support for video input/output in various formats

## Prerequisites

- requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ro-han12/Football-stats.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your input video in the `input_videos` directory
2. Run the main script:
```bash
python main.py
```
4. create a 'input_videos' folder in root maindirectory & place in your source videos
5. create a 'output_videos' folder in root directory
7. create a stubs folder in the root directory (To store tracking on the source for future enhancements)
6. The processed video will be saved in the `output_videos` directory

## Project Structure

- `main.py`: Main script for video processing
- `utils.py`: Utility functions for video handling
- `trackers.py`: Object tracking implementation
- `models/`: Directory containing YOLOv8 model files
- `input_videos/`: Directory for input videos
- `output_videos/`: Directory for processed videos
- `stubs/`: Directory for tracking stubs

