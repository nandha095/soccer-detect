from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
                    tracks[object][frame_num][track_id]['position_adjusted'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw a smaller ellipse with thinner lines
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width * 0.8), int(0.3*width)), # Smaller ellipse
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=1,  # Thinner lines
            lineType=cv2.LINE_AA  # Anti-aliased for smoother appearance
        )

        # If no track_id is provided, no need to draw the player ID
        if track_id is None:
            return frame

        # Determine visual density around this player (check for nearby players)
        is_cluttered = False
        if hasattr(self, 'last_player_positions'):
            player_pos = (x_center, y2)
            # Check if there are nearby players
            for pos in self.last_player_positions:
                if pos != player_pos:  # Don't compare with itself
                    distance = ((pos[0] - player_pos[0])**2 + (pos[1] - player_pos[1])**2)**0.5
                    if distance < width * 1.5:  # Players are close
                        is_cluttered = True
                        break

        # Create a cleaner and more readable player ID display
        rectangle_width = 24  # Smaller rectangle
        rectangle_height = 16
        
        # Adjust ID position based on clustering
        offset_y = 0
        if is_cluttered:
            # Stagger the IDs vertically in clustered areas
            # Create a hash of the track_id to determine vertical offset
            offset_y = (track_id % 3) * 20 - 20  # -20, 0, or 20 pixels offset

        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15 + offset_y
        y2_rect = (y2 + rectangle_height//2) + 15 + offset_y

        # Semi-transparent background for better readability
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (int(x1_rect),int(y1_rect)),
                      (int(x2_rect),int(y2_rect)),
                      color,
                      cv2.FILLED)
        
        # Apply transparency effect
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        # Calculate text position
        x1_text = x1_rect + 7
        if track_id > 99:
            x1_text -= 7
        elif track_id > 9:
            x1_text -= 3
        
        # Add text with smaller font and better outline for readability
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text),int(y1_rect+12)),  # Adjusted position
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,  # Smaller font
            (0,0,0),
            1  # Thinner text
        )

        return frame

    def draw_triangle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        # Make ball indicator more visible with a glow effect
        # First draw a larger background triangle for the glow effect
        glow_size = 15
        glow_points = np.array([
            [x, y],
            [x-glow_size, y-glow_size*1.5],
            [x+glow_size, y-glow_size*1.5],
        ], dtype=np.int32)
        
        # Semi-transparent glow
        overlay = frame.copy()
        cv2.drawContours(overlay, [glow_points], 0, (255,255,255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw the main triangle
        triangle_size = 10  # Slightly smaller than before
        triangle_points = np.array([
            [x, y],
            [x-triangle_size, y-triangle_size*1.5],
            [x+triangle_size, y-triangle_size*1.5],
        ], dtype=np.int32)
        
        # Fill with main color
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        
        # Add thin border
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 1, cv2.LINE_AA)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total_frames = team_1_num_frames + team_2_num_frames
        
        if total_frames > 0:
            team_1 = team_1_num_frames/total_frames
            team_2 = team_2_num_frames/total_frames
        else:
            team_1 = team_2 = 0

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_ball_paths(self, frame, tracks, frame_num, path_history=15):
        """Draw paths showing ball movement between players."""
        # Get recent frames to show path history - increased from 10 to 15
        start_frame = max(0, frame_num - path_history)
        
        # Collect recent ball positions and player assignments
        ball_positions = []
        for i in range(start_frame, frame_num + 1):
            if i < len(tracks["ball"]) and tracks["ball"][i] and 1 in tracks["ball"][i]:
                ball_pos = get_center_of_bbox(tracks["ball"][i][1]["bbox"])
                ball_positions.append((ball_pos, i))
        
        # Draw path lines
        if len(ball_positions) > 1:
            # Create a separate overlay for the paths to control overall transparency
            path_overlay = frame.copy()
            
            for i in range(len(ball_positions) - 1):
                start_pos, start_frame = ball_positions[i]
                end_pos, end_frame = ball_positions[i+1]
                
                start_point = (int(start_pos[0]), int(start_pos[1]))
                end_point = (int(end_pos[0]), int(end_pos[1]))
                
                # Calculate movement distance
                movement = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                
                # Calculate alpha for fade effect - stronger fade for older points
                recency = (i + 1) / len(ball_positions)
                
                # Reduced threshold to show more movements
                if movement < 3:  # Was 5, now 3
                    continue
                
                # Use different colors based on movement speed - brighter colors
                if movement > 40:  # Fast movement (likely a pass) - reduced from 50 to 40
                    color = (0, 200, 255)  # Brighter orange for passes
                    thickness = int(max(2, 3 * recency))  # Thicker lines
                else:
                    color = (0, 255, 100)  # Brighter green for slower movement
                    thickness = int(max(1, 2 * recency))  # Slightly thicker
                
                # Draw line with anti-aliasing for smoother appearance
                if thickness > 0:
                    # Draw a glow effect around the path for better visibility
                    # First draw a wider, darker line
                    glow_thickness = thickness + 2
                    cv2.line(path_overlay, start_point, end_point, (0, 0, 0), glow_thickness, cv2.LINE_AA)
                    
                    # Then draw the colored line on top
                    cv2.line(path_overlay, start_point, end_point, color, thickness, cv2.LINE_AA)
                    
                    # Draw arrow for direction - more visible and on more movements
                    if movement > 20 and recency > 0.4:  # Reduced from 30 to 20, and from 0.5 to 0.4
                        angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
                        arrow_length = 20  # Increased from 15 to 20
                        arrow_angle = np.pi/6  # 30 degrees
                        
                        # Calculate arrow points
                        p1 = (int(end_point[0] - arrow_length * np.cos(angle + arrow_angle)),
                              int(end_point[1] - arrow_length * np.sin(angle + arrow_angle)))
                        p2 = (int(end_point[0] - arrow_length * np.cos(angle - arrow_angle)),
                              int(end_point[1] - arrow_length * np.sin(angle - arrow_angle)))
                        
                        # Draw glow effect for arrows too
                        cv2.line(path_overlay, end_point, p1, (0, 0, 0), thickness+1, cv2.LINE_AA)
                        cv2.line(path_overlay, end_point, p2, (0, 0, 0), thickness+1, cv2.LINE_AA)
                        
                        # Draw the colored arrows
                        cv2.line(path_overlay, end_point, p1, color, thickness, cv2.LINE_AA)
                        cv2.line(path_overlay, end_point, p2, color, thickness, cv2.LINE_AA)
            
            # Apply the path overlay with higher minimum opacity
            player_count = len(tracks["players"][frame_num])
            # Increased minimum from 0.3 to 0.5, and maximum from 0.7 to 0.8
            path_alpha = max(0.5, 0.8 - (player_count * 0.015))  # Slower reduction in opacity with player count
            
            cv2.addWeighted(path_overlay, path_alpha, frame, 1 - path_alpha, 0, frame)
            
            # Add visual markers at ball positions for better visibility
            for i, (pos, _) in enumerate(ball_positions):
                # Only show markers for every third position to avoid clutter
                if i % 3 == 0:
                    marker_size = int(3 + 2 * (i / len(ball_positions)))  # Bigger markers for recent positions
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), marker_size, (0, 255, 255), -1, cv2.LINE_AA)
        
        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Keep track of player positions to detect clustering
            self.last_player_positions = []
            for track_id, player in player_dict.items():
                if "bbox" in player:
                    x_center, _ = get_center_of_bbox(player["bbox"])
                    y2 = int(player["bbox"][3])
                    self.last_player_positions.append((x_center, y2))

            # Always use full path history regardless of player count
            frame = self.draw_ball_paths(frame, tracks, frame_num, path_history=15)

            # Draw Players
            for track_id, player in player_dict.items():
                if "bbox" not in player:
                    continue
                    
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                if "bbox" not in referee:
                    continue
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                if "bbox" not in ball:
                    continue
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))

            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames