import sys
sys.path.append('../')
import numpy as np
import cv2
from utils import get_center_of_bbox, measure_distance

class PassDetector:
    def __init__(self):
        # Parameters for pass detection
        self.min_pass_distance = 50  # Minimum distance (pixels) to consider as a pass
        self.min_pass_speed = 30      # Minimum speed to consider a pass (faster than dribbling)
        self.max_receiving_time = 20  # Maximum frames to wait for pass reception
        self.min_receiving_distance = 30  # Minimum distance for a player to be considered a receiver
        self.player_possession_threshold = 70  # Maximum distance to consider player has ball possession
        self.pass_events = []         # Store detected pass events
    
    def detect_passes(self, video_frames, tracks, team_ball_control):
        """Detect passing events between players."""
        print("Detecting passing events...")
        self.pass_events = []
        
        ongoing_pass = None
        
        # Process each frame to detect passes
        for frame_num in range(3, len(tracks['players'])):
            # Skip if ball not detected in current or previous frames
            if not self._check_ball_detected(tracks, frame_num, 3):
                continue
            
            # Find player with ball in current frame
            ball_pos = self._get_ball_position(tracks, frame_num)
            player_with_ball = self._find_player_with_ball(tracks['players'][frame_num], ball_pos)
            
            # If we have an ongoing pass, check if it's complete
            if ongoing_pass is not None:
                if self._check_pass_completion(ongoing_pass, frame_num, tracks, player_with_ball):
                    self.pass_events.append(ongoing_pass)
                    ongoing_pass = None
                elif frame_num - ongoing_pass['start_frame'] > self.max_receiving_time:
                    # Pass timed out without completion
                    ongoing_pass['successful'] = False
                    ongoing_pass['end_frame'] = frame_num
                    self.pass_events.append(ongoing_pass)
                    ongoing_pass = None
            
            # Look for new pass only if no ongoing pass
            if ongoing_pass is None:
                # Check if ball is moving fast enough to be a pass
                ball_speed, ball_vector = self._calculate_ball_speed_and_vector(tracks, frame_num)
                
                if (ball_speed > self.min_pass_speed and
                    player_with_ball != -1):
                    
                    # Get potential receivers (teammates) in ball trajectory
                    potential_receivers = self._find_potential_receivers(
                        tracks, frame_num, ball_pos, ball_vector, 
                        tracks['players'][frame_num][player_with_ball].get('team', 0)
                    )
                    
                    if potential_receivers:
                        # Register new pass
                        ongoing_pass = {
                            'start_frame': frame_num,
                            'sender_id': player_with_ball,
                            'sender_team': tracks['players'][frame_num][player_with_ball].get('team', 0),
                            'sender_position': tracks['players'][frame_num][player_with_ball]['position'],
                            'initial_ball_position': ball_pos,
                            'initial_ball_speed': ball_speed,
                            'potential_receivers': potential_receivers,
                            'receiver_id': None,
                            'receiver_position': None,
                            'successful': None,
                            'pass_distance': None,
                            'end_frame': None,
                            'pass_type': self._determine_pass_type(ball_speed, ball_vector)
                        }
        
        return self.pass_events
    
    def _check_ball_detected(self, tracks, frame_num, lookback=1):
        """Check if ball is detected in current and previous frames."""
        for i in range(frame_num - lookback + 1, frame_num + 1):
            if (i < 0 or i >= len(tracks['ball']) or 
                1 not in tracks['ball'][i] or 
                'bbox' not in tracks['ball'][i][1]):
                return False
        return True
    
    def _get_ball_position(self, tracks, frame_num):
        """Get ball position in specified frame."""
        if (frame_num < len(tracks['ball']) and 
            1 in tracks['ball'][frame_num] and 
            'bbox' in tracks['ball'][frame_num][1]):
            return get_center_of_bbox(tracks['ball'][frame_num][1]['bbox'])
        return None
    
    def _find_player_with_ball(self, players, ball_position):
        """Find player closest to the ball within possession threshold."""
        min_distance = self.player_possession_threshold
        player_with_ball = -1
        
        for player_id, player in players.items():
            if 'position' not in player:
                continue
                
            player_pos = player['position']
            distance = measure_distance(player_pos, ball_position)
            
            if distance < min_distance:
                min_distance = distance
                player_with_ball = player_id
                
        return player_with_ball
    
    def _calculate_ball_speed_and_vector(self, tracks, frame_num):
        """Calculate ball speed and movement vector between consecutive frames."""
        curr_ball_pos = self._get_ball_position(tracks, frame_num)
        prev_ball_pos = self._get_ball_position(tracks, frame_num - 1)
        
        if curr_ball_pos is None or prev_ball_pos is None:
            return 0, (0, 0)
        
        dx = curr_ball_pos[0] - prev_ball_pos[0]
        dy = curr_ball_pos[1] - prev_ball_pos[1]
        
        speed = np.sqrt(dx**2 + dy**2)
        vector = (dx, dy)
        
        return speed, vector
    
    def _find_potential_receivers(self, tracks, frame_num, ball_pos, ball_vector, sender_team):
        """Find potential pass receivers based on ball trajectory."""
        potential_receivers = []
        
        # Create a unit vector for the ball direction
        vector_magnitude = np.sqrt(ball_vector[0]**2 + ball_vector[1]**2)
        if vector_magnitude < 1e-6:  # Avoid division by zero
            return potential_receivers
            
        unit_vector = (ball_vector[0] / vector_magnitude, ball_vector[1] / vector_magnitude)
        
        # Loop through players to find potential receivers
        for player_id, player in tracks['players'][frame_num].items():
            if 'position' not in player or player.get('team', 0) != sender_team:
                continue
                
            player_pos = player['position']
            
            # Vector from ball to player
            to_player = (player_pos[0] - ball_pos[0], player_pos[1] - ball_pos[1])
            to_player_magnitude = np.sqrt(to_player[0]**2 + to_player[1]**2)
            
            if to_player_magnitude < self.min_receiving_distance:
                continue  # Player is too close to be a receiver
                
            # Calculate dot product to see if player is in the direction of the ball
            dot_product = to_player[0] * unit_vector[0] + to_player[1] * unit_vector[1]
            
            # Check if player is in front of the ball (in direction of movement)
            if dot_product > 0:
                # Calculate the projection of player onto the ball direction vector
                projection = dot_product / vector_magnitude
                
                # Calculate perpendicular distance from player to ball trajectory
                perpendicular = np.sqrt(to_player_magnitude**2 - projection**2)
                
                # Add as potential receiver if close enough to trajectory
                if perpendicular < 150:  # Threshold for potential receiver
                    potential_receivers.append({
                        'player_id': player_id,
                        'position': player_pos,
                        'distance': to_player_magnitude,
                        'projection': projection,
                        'perpendicular': perpendicular
                    })
        
        # Sort by projection distance (closest first)
        potential_receivers.sort(key=lambda x: x['projection'])
        
        return potential_receivers
    
    def _check_pass_completion(self, pass_event, frame_num, tracks, current_player_with_ball):
        """Check if a pass has been completed."""
        # Get current ball position
        ball_pos = self._get_ball_position(tracks, frame_num)
        if ball_pos is None:
            return False
            
        # Calculate total ball travel distance
        initial_pos = pass_event['initial_ball_position']
        total_distance = measure_distance(initial_pos, ball_pos)
        
        # Check if current player with ball is one of the potential receivers
        receiver_found = False
        if current_player_with_ball != -1 and current_player_with_ball != pass_event['sender_id']:
            # Check if current player is in the same team (successful pass)
            if current_player_with_ball in tracks['players'][frame_num]:
                player_team = tracks['players'][frame_num][current_player_with_ball].get('team', 0)
                
                # Check if player is in potential receivers list
                for receiver in pass_event['potential_receivers']:
                    if receiver['player_id'] == current_player_with_ball:
                        receiver_found = True
                        receiver_position = tracks['players'][frame_num][current_player_with_ball]['position']
                        
                        # Update pass event with completion info
                        pass_event['receiver_id'] = current_player_with_ball
                        pass_event['receiver_position'] = receiver_position
                        pass_event['successful'] = (player_team == pass_event['sender_team'])
                        pass_event['pass_distance'] = total_distance
                        pass_event['end_frame'] = frame_num
                        return True
        
        # If no receiver found but ball has traveled far, consider the pass as an attempt
        if not receiver_found and total_distance > self.min_pass_distance:
            if frame_num - pass_event['start_frame'] >= self.max_receiving_time // 2:
                # Likely an unsuccessful pass (no receiver reached the ball)
                pass_event['successful'] = False
                pass_event['pass_distance'] = total_distance
                pass_event['end_frame'] = frame_num
                return True
                
        return False
    
    def _determine_pass_type(self, ball_speed, ball_vector):
        """Determine the type of pass based on its characteristics."""
        # Calculate horizontal and vertical components
        dx, dy = ball_vector
        horizontal_component = abs(dx)
        vertical_component = abs(dy)
        
        # Determine pass type based on speed and direction
        if ball_speed > 60:
            if horizontal_component > 2 * vertical_component:
                return "long_ground_pass"  # Fast and mostly horizontal
            else:
                return "long_aerial_pass"  # Fast with significant vertical component
        else:
            if horizontal_component > 2 * vertical_component:
                return "short_ground_pass"  # Slower and mostly horizontal
            else:
                return "short_aerial_pass"  # Slower with significant vertical component
    
    def get_pass_stats(self):
        """Generate summary statistics for pass events."""
        if not self.pass_events:
            return {
                "total_passes": 0,
                "successful_passes": 0,
                "unsuccessful_passes": 0,
                "success_rate": 0,
                "pass_types": {
                    "long_ground_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                    "long_aerial_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                    "short_ground_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                    "short_aerial_pass": {"total": 0, "successful": 0, "unsuccessful": 0}
                },
                "team_stats": {
                    1: {"total": 0, "successful": 0, "unsuccessful": 0, "success_rate": 0},
                    2: {"total": 0, "successful": 0, "unsuccessful": 0, "success_rate": 0}
                }
            }
            
        # Initialize stats
        stats = {
            "total_passes": len(self.pass_events),
            "successful_passes": 0,
            "unsuccessful_passes": 0,
            "success_rate": 0,
            "pass_types": {
                "long_ground_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                "long_aerial_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                "short_ground_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                "short_aerial_pass": {"total": 0, "successful": 0, "unsuccessful": 0}
            },
            "team_stats": {
                1: {"total": 0, "successful": 0, "unsuccessful": 0, "success_rate": 0},
                2: {"total": 0, "successful": 0, "unsuccessful": 0, "success_rate": 0}
            }
        }
        
        # Calculate stats from pass events
        for pass_event in self.pass_events:
            if pass_event['successful'] is True:
                stats["successful_passes"] += 1
            else:
                stats["unsuccessful_passes"] += 1
                
            # Track pass types
            pass_type = pass_event.get('pass_type', 'short_ground_pass')
            stats["pass_types"][pass_type]["total"] += 1
            
            if pass_event['successful'] is True:
                stats["pass_types"][pass_type]["successful"] += 1
            else:
                stats["pass_types"][pass_type]["unsuccessful"] += 1
                
            # Track team stats
            team = pass_event.get('sender_team', 0)
            if team in [1, 2]:
                stats["team_stats"][team]["total"] += 1
                if pass_event['successful'] is True:
                    stats["team_stats"][team]["successful"] += 1
                else:
                    stats["team_stats"][team]["unsuccessful"] += 1
        
        # Calculate success rates
        if stats["total_passes"] > 0:
            stats["success_rate"] = stats["successful_passes"] / stats["total_passes"] * 100
            
        for team in [1, 2]:
            if stats["team_stats"][team]["total"] > 0:
                stats["team_stats"][team]["success_rate"] = (
                    stats["team_stats"][team]["successful"] / 
                    stats["team_stats"][team]["total"] * 100
                )
                
        return stats 