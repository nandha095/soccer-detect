import sys
sys.path.append('../')
import numpy as np
import cv2
from utils import get_center_of_bbox, measure_distance

class ShotDetector:
    def __init__(self):
        # Parameters for shot detection
        self.shot_speed_threshold = 30  # REDUCED: Minimum speed to consider as a shot (was 40)
        self.close_shot_distance = 300  # INCREASED: Distance (in pixels) threshold for close shots (was 250)
        self.goal_line_y_range = (180, 400)  # Y-coordinate range of goal line (adjust based on pitch perspective)
        self.goal_line_x_range = (900, 1000)  # X-coordinate range of goal line (adjust based on pitch orientation)
        self.shot_detection_cooldown = 25  # REDUCED: Frames to wait before detecting another shot (was 30)
        self.shot_cooldown_counter = 0
        self.recent_shots = []  # To keep track of recent shots
        
    def detect_goals(self, video_frames, tracks):
        """Detect goals and shots in the video based on ball trajectory."""
        print("Detecting shots and goals...")
        shot_events = []
        last_shot_frame = -100  # To prevent multiple detections of the same shot
        
        for frame_num in range(1, len(tracks['ball'])):
            # Skip if we're in cooldown period
            if frame_num - last_shot_frame < self.shot_detection_cooldown:
                continue
                
            # Skip if ball not detected in current or previous frame
            if (1 not in tracks['ball'][frame_num] or 
                'bbox' not in tracks['ball'][frame_num][1] or
                1 not in tracks['ball'][frame_num-1] or 
                'bbox' not in tracks['ball'][frame_num-1][1]):
                continue
            
            # Get current and previous ball positions
            curr_ball_bbox = tracks['ball'][frame_num][1]['bbox']
            prev_ball_bbox = tracks['ball'][frame_num-1][1]['bbox']
            
            curr_ball_pos = get_center_of_bbox(curr_ball_bbox)
            prev_ball_pos = get_center_of_bbox(prev_ball_bbox)
            
            # Calculate ball movement vector and speed
            ball_vector = (curr_ball_pos[0] - prev_ball_pos[0], curr_ball_pos[1] - prev_ball_pos[1])
            ball_speed = np.sqrt(ball_vector[0]**2 + ball_vector[1]**2)
            
            # Check if the ball is moving fast enough to be a potential shot
            if ball_speed > self.shot_speed_threshold:
                # Check direction toward goal (assuming goal is on the right side of the field)
                is_toward_goal = ball_vector[0] > 0
                
                if is_toward_goal:
                    # Check if shot is heading toward goal area
                    is_on_target = self._is_on_target(curr_ball_pos, ball_vector)
                    
                    # Determine shot distance
                    distance_to_goal = self._calculate_distance_to_goal(curr_ball_pos)
                    shot_type = "close" if distance_to_goal < self.close_shot_distance else "long"
                    
                    # Check if ball crosses goal line in next few frames
                    is_successful = self._check_goal_scored(tracks['ball'], frame_num)
                    
                    # Record shot
                    shot_event = {
                        "frame": frame_num,
                        "position": curr_ball_pos,
                        "type": shot_type,
                        "successful": is_successful,
                        "on_target": is_on_target,
                        "distance_to_goal": distance_to_goal,
                        "speed": ball_speed
                    }
                    
                    shot_events.append(shot_event)
                    last_shot_frame = frame_num
                    
                    # Find player who took the shot
                    shooter_id = self._find_shooter(tracks['players'][frame_num], curr_ball_pos)
                    if shooter_id != -1:
                        shot_event["player_id"] = shooter_id
                        if shooter_id in tracks['players'][frame_num] and 'team' in tracks['players'][frame_num][shooter_id]:
                            shot_event["team"] = tracks['players'][frame_num][shooter_id]['team']
        
        self.recent_shots = shot_events
        return shot_events
    
    def _is_on_target(self, ball_pos, ball_vector):
        """Check if the shot is on target (heading toward goal)."""
        # Project the ball's path and see if it intersects with goal
        future_x = ball_pos[0] + ball_vector[0] * 10  # Project forward
        future_y = ball_pos[1] + ball_vector[1] * 10
        
        # Check if projected position is within goal area
        return (self.goal_line_x_range[0] <= future_x <= self.goal_line_x_range[1] and
                self.goal_line_y_range[0] <= future_y <= self.goal_line_y_range[1])
    
    def _calculate_distance_to_goal(self, ball_pos):
        """Calculate distance from ball to goal center."""
        goal_center_x = (self.goal_line_x_range[0] + self.goal_line_x_range[1]) / 2
        goal_center_y = (self.goal_line_y_range[0] + self.goal_line_y_range[1]) / 2
        goal_center = (goal_center_x, goal_center_y)
        
        return measure_distance(ball_pos, goal_center)
    
    def _check_goal_scored(self, ball_tracks, frame_num):
        """Check if ball crosses goal line in the next few frames."""
        for i in range(frame_num, min(frame_num + 15, len(ball_tracks))):
            if 1 in ball_tracks[i] and 'bbox' in ball_tracks[i][1]:
                ball_pos = get_center_of_bbox(ball_tracks[i][1]['bbox'])
                
                # Check if ball is in goal area
                if (self.goal_line_x_range[0] <= ball_pos[0] <= self.goal_line_x_range[1] and
                    self.goal_line_y_range[0] <= ball_pos[1] <= self.goal_line_y_range[1]):
                    return True
        
        return False
    
    def _find_shooter(self, players, ball_pos):
        """Find the player closest to the ball who likely took the shot."""
        min_distance = 100  # Maximum distance to consider
        shooter_id = -1
        
        for player_id, player in players.items():
            if 'bbox' not in player:
                continue
                
            player_pos = get_center_of_bbox(player['bbox'])
            distance = measure_distance(player_pos, ball_pos)
            
            if distance < min_distance:
                min_distance = distance
                shooter_id = player_id
                
        return shooter_id
    
    def draw_shot_annotations(self, video_frames, shot_events):
        """Draw shot information on video frames."""
        for shot in shot_events:
            frame_num = shot["frame"]
            if frame_num < len(video_frames):
                frame = video_frames[frame_num]
                
                # Draw shot indicator
                position = shot["position"]
                success_color = (0, 255, 0) if shot["successful"] else (0, 0, 255)  # Green for goal, red for miss
                shot_type_text = "Close" if shot["type"] == "close" else "Long"
                result_text = "GOAL!" if shot["successful"] else "Miss"
                
                # Draw circle at shot position
                cv2.circle(frame, (int(position[0]), int(position[1])), 15, success_color, 2)
                
                # Draw shot info text
                info_text = f"{shot_type_text} Shot: {result_text}"
                cv2.putText(frame, info_text, 
                            (int(position[0]) - 100, int(position[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, success_color, 2)
                
                # Draw projected shot path
                if "player_id" in shot:
                    # Draw line from player to shot position
                    cv2.line(frame, 
                            (int(position[0]), int(position[1])),
                            (int(self.goal_line_x_range[0]), int((self.goal_line_y_range[0] + self.goal_line_y_range[1]) / 2)),
                            success_color, 2)
        
        # Add shot statistics to all frames in the video
        if shot_events:
            self.recent_shots = shot_events
            for frame in video_frames:
                self.draw_shot_stats(frame)
                
        return video_frames
    
    def draw_shot_stats(self, frame):
        """Draw shot statistics on the frame."""
        stats = self.summarize_shot_stats()
        
        # Draw semi-transparent rectangle for shot stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (400, 250), (255, 255, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw title
        cv2.putText(frame, "SHOT STATISTICS", (75, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw team shots
        y_pos = 110
        for team_id in [1, 2]:
            team_text = f"Team {team_id}"
            cv2.putText(frame, team_text, (75, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_pos += 25
            
            # Close shots
            close_successful = stats["team_shots"][team_id]["close"]["successful"]
            close_unsuccessful = stats["team_shots"][team_id]["close"]["unsuccessful"]
            close_text = f"  Close: {close_successful} Goals, {close_unsuccessful} Misses"
            cv2.putText(frame, close_text, (75, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 25
            
            # Long shots
            long_successful = stats["team_shots"][team_id]["long"]["successful"]
            long_unsuccessful = stats["team_shots"][team_id]["long"]["unsuccessful"]
            long_text = f"  Long: {long_successful} Goals, {long_unsuccessful} Misses"
            cv2.putText(frame, long_text, (75, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 25
        
        # Draw totals
        total_shots = stats["total_shots"]
        total_goals = stats["close_shots"]["successful"] + stats["long_shots"]["successful"]
        cv2.putText(frame, f"Total Shots: {total_shots}", (75, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_pos += 25
        cv2.putText(frame, f"Total Goals: {total_goals}", (75, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def summarize_shot_stats(self):
        """Generate summary of shot statistics."""
        stats = {
            "close_shots": {"successful": 0, "unsuccessful": 0, "total": 0},
            "long_shots": {"successful": 0, "unsuccessful": 0, "total": 0},
            "team_shots": {1: {"close": {"successful": 0, "unsuccessful": 0}, 
                              "long": {"successful": 0, "unsuccessful": 0}},
                          2: {"close": {"successful": 0, "unsuccessful": 0}, 
                              "long": {"successful": 0, "unsuccessful": 0}}},
            "total_shots": 0
        }
        
        for shot in self.recent_shots:
            shot_type = shot["type"]
            successful = shot["successful"]
            team = shot.get("team", 0)
            
            # Update overall stats
            if shot_type == "close":
                stats["close_shots"]["total"] += 1
                if successful:
                    stats["close_shots"]["successful"] += 1
                else:
                    stats["close_shots"]["unsuccessful"] += 1
            else:  # long shot
                stats["long_shots"]["total"] += 1
                if successful:
                    stats["long_shots"]["successful"] += 1
                else:
                    stats["long_shots"]["unsuccessful"] += 1
            
            # Update team-specific stats
            if team in [1, 2]:
                if shot_type == "close":
                    if successful:
                        stats["team_shots"][team]["close"]["successful"] += 1
                    else:
                        stats["team_shots"][team]["close"]["unsuccessful"] += 1
                else:  # long shot
                    if successful:
                        stats["team_shots"][team]["long"]["successful"] += 1
                    else:
                        stats["team_shots"][team]["long"]["unsuccessful"] += 1
            
            stats["total_shots"] += 1
        
        return stats 