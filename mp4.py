import subprocess
import os

def convert_avi_to_mp4(input_file, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".mp4"
    
    command = [
        "ffmpeg",
        "-i", input_file,  # Input file
        "-c:v", "libx264",  # Video codec
        "-preset", "fast",  # Encoding speed
        "-crf", "23",       # Quality (lower = better)
        "-c:a", "aac",      # Audio codec
        "-b:a", "192k",     # Audio bitrate
        "-movflags", "+faststart",  # Optimize for web streaming
        output_file
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

# Example usage:
convert_avi_to_mp4("output_videos/output_video.avi", "output.mp4")
