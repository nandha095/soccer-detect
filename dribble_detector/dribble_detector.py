import sys
sys.path.append('../')
import numpy as np
import cv2
from utils import get_center_of_bbox, measure_distance

class DribbleDetector:
    def __init__(self):
        # Parameters for dribble detection
        self.min_dribble_duration = 8   # REDUCED: Minimum frames to consider a dribbling action (was 10)
        self.player_proximity_threshold = 90  # INCREASED: Distance in pixels to consider player-to-player interaction (was 70)
        self.direction_change_threshold = 20  # REDUCED: Threshold for detecting change in direction (degrees) (was 25)
        self.speed_burst_threshold = 10  # REDUCED: Threshold for detecting speed burst (for special dribbles) (was 15)
        self.possession_change_threshold = 90  # INCREASED: Distance threshold to detect possession change (was 80)
        self.successful_continuation_frames = 15  # REDUCED: Frames to check for continued possession after dribble (was 20)
        self.dribble_events = []  # Store detected dribble events
        self.ongoing_dribbles = {}  # Track ongoing dribble attempts
        self.player_directions = {}  # Store recent player directions for each tracked player
        self.player_speeds = {}  # Store recent player speeds
    
    def detect_dribbles(self, video_frames, tracks, team_ball_control):
        """Detect various dribbling scenarios in the football match."""
        print("Detecting dribbling actions...")
        self.dribble_events = []
        self.player_directions = {}
        self.player_speeds = {}
        self.ongoing_dribbles = {}
        
        # Process each frame to detect dribbles
        for frame_num in range(1, len(tracks['players'])):
            # Skip if ball not detected
            if (frame_num >= len(tracks['ball']) or 
                1 not in tracks['ball'][frame_num] or 
                'bbox' not in tracks['ball'][frame_num][1]):
                continue
            
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            ball_position = get_center_of_bbox(ball_bbox)
            
            # Find player with ball possession
            player_with_ball = self._find_player_with_ball(tracks['players'][frame_num], ball_position)
            
            if player_with_ball != -1:
                # Check for dribbling scenarios
                self._process_dribbling_scenarios(frame_num, tracks, player_with_ball, ball_position)
                
                # Check for completion of ongoing dribbles
                self._check_dribble_completion(frame_num, tracks, player_with_ball)
            
            # Update player tracking data
            self._update_player_tracking_data(frame_num, tracks)
        
        # Finalize any ongoing dribbles
        self._finalize_ongoing_dribbles(len(tracks['players']), tracks)
        
        return self.dribble_events
    
    def _find_player_with_ball(self, players, ball_position):
        """Find player closest to the ball who likely has possession."""
        min_distance = 60  # Maximum distance to consider for possession
        player_with_ball = -1
        
        for player_id, player in players.items():
            if 'bbox' not in player or 'position' not in player:
                continue
                
            player_pos = player['position']
            distance = measure_distance(player_pos, ball_position)
            
            if distance < min_distance:
                min_distance = distance
                player_with_ball = player_id
                
        return player_with_ball
    
    def _process_dribbling_scenarios(self, frame_num, tracks, player_with_ball, ball_position):
        """
        Process current frame to identify dribbling scenarios:
        
        1. Regular Dribble: A dribble with 3+ opponents nearby
           Context: Often in crowded midfield or near the box where space is tight
           
        2. 1v1 Dribble: A dribble against exactly one opponent
           Context: Common on the wings or in attacking duels
           
        3. 1v2 Dribble: A dribble against exactly two opponents
           Context: High-risk scenario during counterattacks or breaking defensive lines
           
        4. Special Dribble: A dribble with significant direction change AND speed burst
           Context: Advanced/skillful moves like sharp cut-ins with acceleration
        """
        players = tracks['players'][frame_num]
        
        if player_with_ball not in players or 'position' not in players[player_with_ball]:
            return
        
        dribbler_pos = players[player_with_ball]['position']
        dribbler_team = players[player_with_ball].get('team', 0)
        
        # Find nearby players from opposite team
        nearby_opponents = []
        for player_id, player in players.items():
            if (player_id != player_with_ball and 
                'position' in player and 
                player.get('team', 0) != dribbler_team):
                
                opponent_pos = player['position']
                distance = measure_distance(dribbler_pos, opponent_pos)
                
                if distance < self.player_proximity_threshold:
                    nearby_opponents.append({
                        'id': player_id,
                        'position': opponent_pos,
                        'distance': distance
                    })
        
        # Classify dribbling scenario based on number of nearby opponents
        scenario_type = None
        if len(nearby_opponents) == 1:
            scenario_type = "one_vs_one"  # 1v1 dribble - common on wings or attacking duels
        elif len(nearby_opponents) == 2:
            scenario_type = "one_vs_two"  # 1v2 dribble - high-risk scenario during counterattacks
        elif len(nearby_opponents) >= 3:
            scenario_type = "dribble"  # Regular dribble - in crowded midfield or near the box
        
        if scenario_type:
            # Check if this is a new dribble event
            if player_with_ball not in self.ongoing_dribbles:
                # Start tracking a new dribble
                self.ongoing_dribbles[player_with_ball] = {
                    'start_frame': frame_num,
                    'player_id': player_with_ball,
                    'team': dribbler_team,
                    'type': scenario_type,
                    'start_position': dribbler_pos,
                    'opponents': [opp['id'] for opp in nearby_opponents],
                    'opponent_count': len(nearby_opponents),
                    'direction_changes': 0,
                    'max_speed': 0 if player_with_ball not in self.player_speeds 
                                else max(self.player_speeds[player_with_ball]),
                    'is_special': False
                }
            elif self.ongoing_dribbles[player_with_ball]['type'] != scenario_type:
                # Scenario type changed, update it
                self.ongoing_dribbles[player_with_ball]['type'] = scenario_type
                self.ongoing_dribbles[player_with_ball]['opponents'] = [opp['id'] for opp in nearby_opponents]
                self.ongoing_dribbles[player_with_ball]['opponent_count'] = len(nearby_opponents)
            
            # Check for direction changes and special dribbles
            if player_with_ball in self.player_directions and len(self.player_directions[player_with_ball]) >= 5:
                # Get recent direction changes
                recent_directions = self.player_directions[player_with_ball][-5:]
                max_direction_diff = self._calculate_max_direction_diff(recent_directions)
                
                # Check for significant direction change (> 25 degrees)
                if max_direction_diff > self.direction_change_threshold:
                    self.ongoing_dribbles[player_with_ball]['direction_changes'] += 1
                
                # Check for speed burst
                if player_with_ball in self.player_speeds:
                    recent_speeds = self.player_speeds[player_with_ball][-5:]
                    current_speed = recent_speeds[-1]
                    avg_prev_speed = sum(recent_speeds[:-1]) / len(recent_speeds[:-1]) if len(recent_speeds) > 1 else 0
                    
                    # Update max speed
                    self.ongoing_dribbles[player_with_ball]['max_speed'] = max(
                        self.ongoing_dribbles[player_with_ball]['max_speed'], 
                        current_speed
                    )
                    
                    # Check if this is a special dribble:
                    # Requires BOTH direction change > 25 degrees AND speed burst
                    if (max_direction_diff > self.direction_change_threshold and 
                        current_speed > avg_prev_speed + self.speed_burst_threshold):
                        self.ongoing_dribbles[player_with_ball]['is_special'] = True
    
    def _check_dribble_completion(self, frame_num, tracks, current_player_with_ball):
        """Check if any ongoing dribbles have completed in this frame."""
        completed_dribbles = []
        
        for player_id, dribble_info in self.ongoing_dribbles.items():
            # Check if player lost the ball to someone else
            if player_id != current_player_with_ball:
                # Get last known position of the dribbler
                last_player_pos = None
                last_frame = min(frame_num, len(tracks['players']) - 1)
                
                if (player_id in tracks['players'][last_frame] and 
                    'position' in tracks['players'][last_frame][player_id]):
                    last_player_pos = tracks['players'][last_frame][player_id]['position']
                
                # Get ball position
                ball_pos = None
                if (1 in tracks['ball'][frame_num] and 
                    'bbox' in tracks['ball'][frame_num][1]):
                    ball_pos = get_center_of_bbox(tracks['ball'][frame_num][1]['bbox'])
                
                # If we have positions for both, check if ball moved away
                if last_player_pos and ball_pos:
                    distance = measure_distance(last_player_pos, ball_pos)
                    
                    # If ball is far from player, consider dribble completed
                    if distance > self.possession_change_threshold:
                        dribble_duration = frame_num - dribble_info['start_frame']
                        
                        if dribble_duration >= self.min_dribble_duration:
                            # Determine success/failure based on who has the ball now
                            successful = False
                            
                            # If teammate has the ball, consider it successful
                            if (current_player_with_ball != -1 and 
                                current_player_with_ball in tracks['players'][frame_num] and
                                'team' in tracks['players'][frame_num][current_player_with_ball] and
                                tracks['players'][frame_num][current_player_with_ball]['team'] == dribble_info['team']):
                                successful = True
                            
                            # Create completed dribble event
                            dribble_event = {
                                'type': dribble_info['type'],
                                'start_frame': dribble_info['start_frame'],
                                'end_frame': frame_num,
                                'player_id': player_id,
                                'team': dribble_info['team'],
                                'successful': successful,
                                'direction_changes': dribble_info['direction_changes'],
                                'is_special': dribble_info['is_special'],
                                'max_speed': dribble_info['max_speed'],
                                'duration': dribble_duration,
                                'opponent_count': dribble_info['opponent_count']
                            }
                            
                            self.dribble_events.append(dribble_event)
                            completed_dribbles.append(player_id)
        
        # Remove completed dribbles
        for player_id in completed_dribbles:
            if player_id in self.ongoing_dribbles:
                del self.ongoing_dribbles[player_id]
    
    def _update_player_tracking_data(self, frame_num, tracks):
        """Update tracked data for player directions and speeds."""
        if frame_num <= 1:
            return
            
        for player_id, player in tracks['players'][frame_num].items():
            # Skip if position not available
            if 'position' not in player:
                continue
                
            # Skip if player wasn't in previous frame
            if (player_id not in tracks['players'][frame_num-1] or
                'position' not in tracks['players'][frame_num-1][player_id]):
                continue
                
            # Get current and previous positions
            curr_pos = player['position']
            prev_pos = tracks['players'][frame_num-1][player_id]['position']
            
            # Calculate movement direction (angle in degrees)
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Store direction
            if player_id not in self.player_directions:
                self.player_directions[player_id] = []
            self.player_directions[player_id].append(angle)
            
            # Limit stored directions to last 10 frames
            if len(self.player_directions[player_id]) > 10:
                self.player_directions[player_id] = self.player_directions[player_id][-10:]
            
            # Store speed if available
            if 'speed' in player:
                if player_id not in self.player_speeds:
                    self.player_speeds[player_id] = []
                self.player_speeds[player_id].append(player['speed'])
                
                # Limit stored speeds to last 10 frames
                if len(self.player_speeds[player_id]) > 10:
                    self.player_speeds[player_id] = self.player_speeds[player_id][-10:]
    
    def _finalize_ongoing_dribbles(self, last_frame, tracks):
        """Finalize any dribbles that are still ongoing at the end of the video."""
        for player_id, dribble_info in list(self.ongoing_dribbles.items()):
            dribble_duration = last_frame - dribble_info['start_frame']
            
            if dribble_duration >= self.min_dribble_duration:
                # Consider these as successful since they lasted until the end
                dribble_event = {
                    'type': dribble_info['type'],
                    'start_frame': dribble_info['start_frame'],
                    'end_frame': last_frame,
                    'player_id': player_id,
                    'team': dribble_info['team'],
                    'successful': True,  # Assumed successful if lasted until end
                    'direction_changes': dribble_info['direction_changes'],
                    'is_special': dribble_info['is_special'],
                    'max_speed': dribble_info['max_speed'],
                    'duration': dribble_duration,
                    'opponent_count': dribble_info['opponent_count']
                }
                
                self.dribble_events.append(dribble_event)
                del self.ongoing_dribbles[player_id]
    
    def _calculate_max_direction_diff(self, directions):
        """Calculate the maximum change in direction from a list of angles."""
        if len(directions) < 2:
            return 0
            
        max_diff = 0
        for i in range(len(directions) - 1):
            diff = abs(directions[i+1] - directions[i])
            # Handle angle wrapping
            if diff > 180:
                diff = 360 - diff
            max_diff = max(max_diff, diff)
            
        return max_diff
    
    def draw_dribble_annotations(self, video_frames, tracks):
        """Draw dribble annotations on video frames."""
        # First pass: add dribble events at their occurrence
        for event in self.dribble_events:
            start_frame = event['start_frame']
            end_frame = event['end_frame']
            player_id = event['player_id']
            
            # Only annotate frames that exist in the video
            for frame_num in range(max(0, start_frame), min(end_frame+1, len(video_frames))):
                if frame_num >= len(video_frames):
                    continue
                    
                frame = video_frames[frame_num]
                
                # Get player position if available
                player_pos = None
                if (frame_num < len(tracks['players']) and 
                    player_id in tracks['players'][frame_num] and 
                    'position' in tracks['players'][frame_num][player_id]):
                    player_pos = tracks['players'][frame_num][player_id]['position']
                
                if player_pos:
                    # Determine event type and color
                    event_type = event['type']
                    is_special = event['is_special']
                    successful = event['successful']
                    
                    # Choose color based on success and type
                    if successful:
                        if is_special:
                            color = (0, 255, 255)  # Yellow for special successful
                        elif event_type == "one_vs_one":
                            color = (0, 255, 0)  # Green for 1v1 successful
                        elif event_type == "one_vs_two":
                            color = (255, 165, 0)  # Orange for 1v2 successful
                        else:
                            color = (255, 0, 255)  # Magenta for other successful
                    else:
                        color = (0, 0, 255)  # Red for unsuccessful
                    
                    # Draw indicator around player
                    cv2.circle(frame, (int(player_pos[0]), int(player_pos[1])), 
                               20, color, 2)
                    
                    # Draw text label
                    label = self._get_event_label(event)
                    cv2.putText(frame, label,
                               (int(player_pos[0]) - 60, int(player_pos[1]) - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add dribble stats to all frames
        for frame in video_frames:
            self._draw_dribble_stats(frame)
            
        return video_frames
    
    def _get_event_label(self, event):
        """Get appropriate label for dribble event."""
        event_type = event['type']
        successful = event['successful']
        is_special = event['is_special']
        
        success_label = "Successful" if successful else "Unsuccessful"
        
        if is_special:
            return f"Special Dribble: {success_label}"
        elif event_type == "one_vs_one":
            return f"1v1: {success_label}"
        elif event_type == "one_vs_two":
            return f"1v2: {success_label}"
        else:
            return f"Dribble: {success_label}"
    
    def _draw_dribble_stats(self, frame):
        """Draw dribble statistics on frame."""
        if not self.dribble_events:
            return frame
            
        # Group dribbles by type and outcome
        stats = {
            "dribble": {"successful": 0, "unsuccessful": 0},
            "one_vs_one": {"successful": 0, "unsuccessful": 0},
            "one_vs_two": {"successful": 0, "unsuccessful": 0},
            "special": {"successful": 0, "unsuccessful": 0},
            "team_stats": {
                1: {"successful": 0, "unsuccessful": 0},
                2: {"successful": 0, "unsuccessful": 0}
            }
        }
        
        for event in self.dribble_events:
            event_type = event['type']
            successful = event['successful']
            is_special = event['is_special']
            team = event.get('team', 0)
            
            success_key = "successful" if successful else "unsuccessful"
            
            # Update type-specific stats
            if is_special:
                stats["special"][success_key] += 1
            else:
                stats[event_type][success_key] += 1
            
            # Update team stats
            if team in [1, 2]:
                stats["team_stats"][team][success_key] += 1
        
        # Draw stats box
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 280), (400, 600), (255, 255, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw title
        cv2.putText(frame, "DRIBBLE STATISTICS", (75, 310), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw type-specific stats
        y_pos = 340
        for event_type, outcomes in [
            ("Regular (3+ opponents)", stats["dribble"]),  # Updated label for clarity
            ("1v1 Dribble", stats["one_vs_one"]), 
            ("1v2 Dribble", stats["one_vs_two"]),
            ("Special Dribble", stats["special"])
        ]:
            total = outcomes["successful"] + outcomes["unsuccessful"]
            success_rate = (outcomes["successful"] / total * 100) if total > 0 else 0
            
            cv2.putText(frame, f"{event_type}:", (75, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_pos += 25
            
            cv2.putText(frame, f"  Successful: {outcomes['successful']}", (75, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 25
            
            cv2.putText(frame, f"  Unsuccessful: {outcomes['unsuccessful']}", (75, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 25
        
        # Draw team stats
        cv2.putText(frame, "Team Success Rates:", (75, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_pos += 25
        
        for team in [1, 2]:
            team_stats = stats["team_stats"][team]
            total = team_stats["successful"] + team_stats["unsuccessful"]
            success_rate = (team_stats["successful"] / total * 100) if total > 0 else 0
            
            cv2.putText(frame, f"  Team {team}: {success_rate:.1f}%", (75, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_pos += 25
        
        return frame
    
    def summarize_dribble_stats(self):
        """Generate summary of dribbling statistics."""
        if not self.dribble_events:
            return {
                "dribble": {"successful": 0, "unsuccessful": 0, "total": 0},
                "one_vs_one": {"successful": 0, "unsuccessful": 0, "total": 0},
                "one_vs_two": {"successful": 0, "unsuccessful": 0, "total": 0},
                "special": {"successful": 0, "unsuccessful": 0, "total": 0},
                "team_stats": {
                    1: {"successful": 0, "unsuccessful": 0, "total": 0, "success_rate": 0},
                    2: {"successful": 0, "unsuccessful": 0, "total": 0, "success_rate": 0}
                },
                "total_dribbles": 0
            }
        
        stats = {
            "dribble": {"successful": 0, "unsuccessful": 0, "total": 0},
            "one_vs_one": {"successful": 0, "unsuccessful": 0, "total": 0},
            "one_vs_two": {"successful": 0, "unsuccessful": 0, "total": 0},
            "special": {"successful": 0, "unsuccessful": 0, "total": 0},
            "team_stats": {
                1: {"successful": 0, "unsuccessful": 0, "total": 0, "success_rate": 0},
                2: {"successful": 0, "unsuccessful": 0, "total": 0, "success_rate": 0}
            },
            "total_dribbles": len(self.dribble_events)
        }
        
        for event in self.dribble_events:
            event_type = event['type']
            successful = event['successful']
            is_special = event['is_special']
            team = event.get('team', 0)
            
            success_key = "successful" if successful else "unsuccessful"
            
            # Update type-specific stats
            if is_special:
                stats["special"][success_key] += 1
                stats["special"]["total"] += 1
            else:
                stats[event_type][success_key] += 1
                stats[event_type]["total"] += 1
            
            # Update team stats
            if team in [1, 2]:
                stats["team_stats"][team][success_key] += 1
                stats["team_stats"][team]["total"] += 1
        
        # Calculate success rates
        for team in [1, 2]:
            team_stats = stats["team_stats"][team]
            if team_stats["total"] > 0:
                team_stats["success_rate"] = team_stats["successful"] / team_stats["total"] * 100
        
        return stats 