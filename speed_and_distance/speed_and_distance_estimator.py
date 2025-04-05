import cv2
import sys 
import numpy as np
sys.path.append('../')
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self):
        self.frame_window = 3     # Reduced from 5 to 3 for more frequent updates
        self.frame_rate = 24
        self.max_interpolation_frames = 5  # Maximum frames to interpolate for missing positions
        self.speed_smoothing_factor = 0.3  # For exponential smoothing of speed
        self.min_movement_threshold = 0.1  # Minimum movement in meters to consider valid

    def interpolate_position(self, tracks, object_type, track_id, frame_num):
        """Interpolate missing positions by looking ahead and behind."""
        # Look for the closest valid positions before and after
        before_pos = None
        after_pos = None
        frames_before = 0
        frames_after = 0

        # Look backward
        for i in range(frame_num-1, max(-1, frame_num-self.max_interpolation_frames-1), -1):
            if (i in tracks[object_type] and 
                track_id in tracks[object_type][i] and 
                'position_transformed' in tracks[object_type][i][track_id] and
                tracks[object_type][i][track_id]['position_transformed'] is not None):
                before_pos = tracks[object_type][i][track_id]['position_transformed']
                frames_before = frame_num - i
                break

        # Look forward
        for i in range(frame_num+1, min(len(tracks[object_type]), frame_num+self.max_interpolation_frames+1)):
            if (i in tracks[object_type] and 
                track_id in tracks[object_type][i] and 
                'position_transformed' in tracks[object_type][i][track_id] and
                tracks[object_type][i][track_id]['position_transformed'] is not None):
                after_pos = tracks[object_type][i][track_id]['position_transformed']
                frames_after = i - frame_num
                break

        # If we found both positions, interpolate between them
        if before_pos is not None and after_pos is not None:
            weight_after = frames_before / (frames_before + frames_after)
            weight_before = 1 - weight_after
            return [
                before_pos[0] * weight_before + after_pos[0] * weight_after,
                before_pos[1] * weight_before + after_pos[1] * weight_after
            ]
        return None

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        last_speeds = {}  # For speed smoothing

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 

            if object not in total_distance:
                total_distance[object] = {}
            if object not in last_speeds:
                last_speeds[object] = {}

            # Initialize all players' total distances
            all_track_ids = set()
            for frame in object_tracks:
                all_track_ids.update(frame.keys())

            for track_id in all_track_ids:
                if track_id not in total_distance[object]:
                    total_distance[object][track_id] = 0
                if track_id not in last_speeds[object]:
                    last_speeds[object][track_id] = 0

            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames - 1):  # Process every frame
                last_frame = min(frame_num + self.frame_window, number_of_frames-1)

                for track_id in all_track_ids:
                    # Get or interpolate start position
                    start_position = None
                    if (track_id in object_tracks[frame_num] and 
                        'position_transformed' in object_tracks[frame_num][track_id]):
                        start_position = object_tracks[frame_num][track_id]['position_transformed']
                    if start_position is None:
                        start_position = self.interpolate_position(tracks, object, track_id, frame_num)

                    # Get or interpolate end position
                    end_position = None
                    if (track_id in object_tracks[last_frame] and 
                        'position_transformed' in object_tracks[last_frame][track_id]):
                        end_position = object_tracks[last_frame][track_id]['position_transformed']
                    if end_position is None:
                        end_position = self.interpolate_position(tracks, object, track_id, last_frame)

                    if start_position is not None and end_position is not None:
                        distance_covered = measure_distance(start_position, end_position)
                        
                        # Only process if movement is above threshold
                        if distance_covered > self.min_movement_threshold:
                            time_elapsed = (last_frame-frame_num)/self.frame_rate
                            speed_meters_per_second = distance_covered/time_elapsed
                            speed_miles_per_hour = speed_meters_per_second * 2.23694

                            # Apply exponential smoothing to speed
                            if track_id in last_speeds[object]:
                                speed_miles_per_hour = (speed_miles_per_hour * self.speed_smoothing_factor + 
                                                      last_speeds[object][track_id] * (1 - self.speed_smoothing_factor))
                            last_speeds[object][track_id] = speed_miles_per_hour

                            distance_miles = distance_covered * 0.000621371
                            total_distance[object][track_id] += distance_miles

                            # Update all frames in the window
                            for frame_num_batch in range(frame_num, last_frame):
                                if track_id not in object_tracks[frame_num_batch]:
                                    continue  # Don't create entries without bbox data
                                object_tracks[frame_num_batch][track_id]['speed'] = speed_miles_per_hour
                                object_tracks[frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
                        else:
                            # If barely moving, set speed to 0 but keep total distance
                            for frame_num_batch in range(frame_num, last_frame):
                                if track_id not in object_tracks[frame_num_batch]:
                                    continue  # Don't create entries without bbox data
                                object_tracks[frame_num_batch][track_id]['speed'] = 0
                                object_tracks[frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self,frames,tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for track_id, track_info in object_tracks[frame_num].items():
                    speed = track_info.get('speed', 0)  # Default to 0 if not available
                    distance = track_info.get('distance', 0)  # Default to 0 if not available
                    
                    if 'bbox' in track_info:  # Only draw if we have position information
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        
                        # Calculate a more compact position that's still near the player
                        # Move text slightly to the right of player to reduce overlap
                        position[0] += 15  # Offset to the right
                        position[1] += 25  # Slightly above foot position

                        position = tuple(map(int,position))
                        
                        # Create a semi-transparent background for better readability
                        text = f"{speed:.1f}mph|{distance:.1f}mi"  # Compact format
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4  # Smaller font
                        thickness = 1  # Thinner text
                        
                        # Get text size for background rectangle
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Draw semi-transparent background
                        overlay = frame.copy()
                        cv2.rectangle(overlay, 
                                    (position[0] - 2, position[1] - text_height - 2),
                                    (position[0] + text_width + 2, position[1] + 2),
                                    (255, 255, 255), 
                                    -1)
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                        
                        # Draw text in black on the semi-transparent background
                        cv2.putText(frame, 
                                  text,
                                  position,
                                  font,
                                  font_scale,
                                  (0, 0, 0),
                                  thickness)

            output_frames.append(frame)
        
        return output_frames