from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from transformation.transformer import ViewTransformer
from speed_and_distance import SpeedAndDistance_Estimator
from shot_detector import ShotDetector
from dribble_detector import DribbleDetector
from pass_detector import PassDetector
import json
import os
from datetime import datetime
import threading
import concurrent.futures
import gc  # For garbage collection

def save_tracking_results(tracks, team_ball_control, shot_events, dribble_events, pass_events, output_path):
    """Save tracking results to a JSON file."""
    # Convert numpy arrays to native Python types
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        else:
            return obj
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "players": {},
        "team_stats": {
            "team1": {
                "total_distance": 0, 
                "avg_speed": 0, 
                "possession_time": 0, 
                "shots": {
                    "close": {"successful": 0, "unsuccessful": 0}, 
                    "long": {"successful": 0, "unsuccessful": 0}
                },
                "dribbles": {
                    "successful": 0, 
                    "unsuccessful": 0, 
                    "dribble": {"successful": 0, "unsuccessful": 0},
                    "one_vs_one": {"successful": 0, "unsuccessful": 0},
                    "one_vs_two": {"successful": 0, "unsuccessful": 0},
                    "special": {"successful": 0, "unsuccessful": 0}
                },
                "passes": {
                    "successful": 0,
                    "unsuccessful": 0,
                    "long_ground_pass": {"successful": 0, "unsuccessful": 0},
                    "long_aerial_pass": {"successful": 0, "unsuccessful": 0},
                    "short_ground_pass": {"successful": 0, "unsuccessful": 0},
                    "short_aerial_pass": {"successful": 0, "unsuccessful": 0}
                }
            },
            "team2": {
                "total_distance": 0, 
                "avg_speed": 0, 
                "possession_time": 0,
                "shots": {
                    "close": {"successful": 0, "unsuccessful": 0}, 
                    "long": {"successful": 0, "unsuccessful": 0}
                },
                "dribbles": {
                    "successful": 0, 
                    "unsuccessful": 0, 
                    "dribble": {"successful": 0, "unsuccessful": 0},
                    "one_vs_one": {"successful": 0, "unsuccessful": 0},
                    "one_vs_two": {"successful": 0, "unsuccessful": 0},
                    "special": {"successful": 0, "unsuccessful": 0}
                },
                "passes": {
                    "successful": 0,
                    "unsuccessful": 0,
                    "long_ground_pass": {"successful": 0, "unsuccessful": 0},
                    "long_aerial_pass": {"successful": 0, "unsuccessful": 0},
                    "short_ground_pass": {"successful": 0, "unsuccessful": 0},
                    "short_aerial_pass": {"successful": 0, "unsuccessful": 0}
                }
            }
        },
        "ball_possession": {
            "team1_frames": int(np.sum(team_ball_control == 1)),
            "team2_frames": int(np.sum(team_ball_control == 2)),
        },
        "shot_stats": {
            "total_shots": len(shot_events),
            "close_shots": {"successful": 0, "unsuccessful": 0},
            "long_shots": {"successful": 0, "unsuccessful": 0}
        },
        "shots": [],
        "dribble_stats": {
            "total_dribbles": len(dribble_events),
            "dribble": {"successful": 0, "unsuccessful": 0},
            "one_vs_one": {"successful": 0, "unsuccessful": 0},
            "one_vs_two": {"successful": 0, "unsuccessful": 0},
            "special": {"successful": 0, "unsuccessful": 0}
        },
        "dribbles": [],
        "pass_stats": {
            "total_passes": len(pass_events),
            "successful_passes": 0,
            "unsuccessful_passes": 0,
            "pass_types": {
                "long_ground_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                "long_aerial_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                "short_ground_pass": {"total": 0, "successful": 0, "unsuccessful": 0},
                "short_aerial_pass": {"total": 0, "successful": 0, "unsuccessful": 0}
            }
        },
        "passes": []
    }

    # Process player data
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track_info in player_track.items():
            player_id_int = int(player_id)
            if player_id_int not in results["players"]:
                results["players"][player_id_int] = {
                    "team": int(track_info.get('team', 0)),
                    "frames_tracked": 0,
                    "total_distance": 0,
                    "max_speed": 0,
                    "avg_speed": 0,
                    "speeds": [],
                    "ball_possession_frames": 0,
                    "shots": {
                        "close": {"successful": 0, "unsuccessful": 0}, 
                        "long": {"successful": 0, "unsuccessful": 0}
                    },
                    "dribbles": {
                        "successful": 0, 
                        "unsuccessful": 0, 
                        "dribble": {"successful": 0, "unsuccessful": 0},
                        "one_vs_one": {"successful": 0, "unsuccessful": 0},
                        "one_vs_two": {"successful": 0, "unsuccessful": 0},
                        "special": {"successful": 0, "unsuccessful": 0}
                    },
                    "passes": {
                        "successful": 0,
                        "unsuccessful": 0,
                        "long_ground_pass": {"successful": 0, "unsuccessful": 0},
                        "long_aerial_pass": {"successful": 0, "unsuccessful": 0},
                        "short_ground_pass": {"successful": 0, "unsuccessful": 0},
                        "short_aerial_pass": {"successful": 0, "unsuccessful": 0}
                    }
                }
            
            player_data = results["players"][player_id_int]
            
            if 'speed' in track_info:
                player_data["frames_tracked"] += 1
                speed = float(track_info['speed'])
                player_data["speeds"].append(speed)
                player_data["max_speed"] = float(max(player_data["max_speed"], speed))
            
            if 'distance' in track_info:
                player_data["total_distance"] = float(track_info['distance'])
            
            if track_info.get('has_ball', False):
                player_data["ball_possession_frames"] += 1

    # Process shot data
    for shot in shot_events:
        # Add shot to results
        shot_entry = {
            "frame": int(shot["frame"]),
            "type": shot["type"],
            "successful": bool(shot["successful"]),
            "distance_to_goal": float(shot["distance_to_goal"]),
            "speed": float(shot["speed"])
        }
        
        # Add player info if available
        if "player_id" in shot:
            player_id = int(shot["player_id"])
            shot_entry["player_id"] = player_id
            
            # Update player shot stats
            if player_id in results["players"]:
                shot_type = shot["type"]
                success_key = "successful" if shot["successful"] else "unsuccessful"
                results["players"][player_id]["shots"][shot_type][success_key] += 1
        
        # Add team info if available
        if "team" in shot:
            team = int(shot["team"])
            shot_entry["team"] = team
            
            # Update team shot stats
            team_key = f"team{team}"
            shot_type = shot["type"]
            success_key = "successful" if shot["successful"] else "unsuccessful"
            results["team_stats"][team_key]["shots"][shot_type][success_key] += 1
        
        # Update overall shot stats
        shot_type = shot["type"]
        success_key = "successful" if shot["successful"] else "unsuccessful"
        results["shot_stats"][f"{shot_type}_shots"][success_key] += 1
        
        results["shots"].append(shot_entry)
    
    # Process dribble data
    for dribble in dribble_events:
        # Add dribble to results
        dribble_entry = {
            "type": dribble["type"],
            "start_frame": int(dribble["start_frame"]),
            "end_frame": int(dribble["end_frame"]),
            "successful": bool(dribble["successful"]),
            "is_special": bool(dribble["is_special"]),
            "duration": int(dribble["duration"]),
            "direction_changes": int(dribble["direction_changes"]),
            "max_speed": float(dribble["max_speed"]),
            "opponent_count": int(dribble.get("opponent_count", 0))
        }
        
        # Add player info if available
        if "player_id" in dribble:
            player_id = int(dribble["player_id"])
            dribble_entry["player_id"] = player_id
            
            # Update player dribble stats
            if player_id in results["players"]:
                dribble_type = dribble["type"]
                success_key = "successful" if dribble["successful"] else "unsuccessful"
                
                # Update overall success/failure count
                results["players"][player_id]["dribbles"][success_key] += 1
                
                # Update specific dribble type
                if dribble["is_special"]:
                    results["players"][player_id]["dribbles"]["special"][success_key] += 1
                else:
                    results["players"][player_id]["dribbles"][dribble_type][success_key] += 1
        
        # Add team info if available
        if "team" in dribble:
            team = int(dribble["team"])
            dribble_entry["team"] = team
            
            # Update team dribble stats
            team_key = f"team{team}"
            dribble_type = dribble["type"]
            success_key = "successful" if dribble["successful"] else "unsuccessful"
            
            # Update overall success/failure count
            results["team_stats"][team_key]["dribbles"][success_key] += 1
            
            # Update specific dribble type
            if dribble["is_special"]:
                results["team_stats"][team_key]["dribbles"]["special"][success_key] += 1
            else:
                results["team_stats"][team_key]["dribbles"][dribble_type][success_key] += 1
        
        # Update overall dribble stats
        dribble_type = "special" if dribble["is_special"] else dribble["type"]
        success_key = "successful" if dribble["successful"] else "unsuccessful"
        results["dribble_stats"][dribble_type][success_key] += 1
        
        results["dribbles"].append(dribble_entry)
        
    # Process pass data
    for pass_event in pass_events:
        # Skip incomplete or invalid passes
        if pass_event['successful'] is None or 'pass_type' not in pass_event:
            continue
            
        # Add pass to results
        pass_entry = {
            "start_frame": int(pass_event["start_frame"]),
            "end_frame": int(pass_event["end_frame"]) if pass_event["end_frame"] is not None else None,
            "pass_type": pass_event["pass_type"],
            "successful": bool(pass_event["successful"]),
            "pass_distance": float(pass_event["pass_distance"]) if pass_event["pass_distance"] is not None else None,
            "initial_ball_speed": float(pass_event["initial_ball_speed"])
        }
        
        # Add sender info
        if "sender_id" in pass_event:
            sender_id = int(pass_event["sender_id"])
            pass_entry["sender_id"] = sender_id
            sender_team = int(pass_event["sender_team"]) if "sender_team" in pass_event else 0
            pass_entry["sender_team"] = sender_team
            
            # Update player pass stats
            if sender_id in results["players"]:
                pass_type = pass_event["pass_type"]
                success_key = "successful" if pass_event["successful"] else "unsuccessful"
                
                # Update overall success/failure count
                results["players"][sender_id]["passes"][success_key] += 1
                
                # Update specific pass type
                results["players"][sender_id]["passes"][pass_type][success_key] += 1
                
            # Update team pass stats
            if sender_team in [1, 2]:
                team_key = f"team{sender_team}"
                pass_type = pass_event["pass_type"]
                success_key = "successful" if pass_event["successful"] else "unsuccessful"
                
                # Update overall success/failure count
                results["team_stats"][team_key]["passes"][success_key] += 1
                
                # Update specific pass type
                results["team_stats"][team_key]["passes"][pass_type][success_key] += 1
        
        # Add receiver info if available
        if "receiver_id" in pass_event and pass_event["receiver_id"] is not None:
            pass_entry["receiver_id"] = int(pass_event["receiver_id"])
        
        # Update overall pass stats
        pass_type = pass_event["pass_type"]
        success_key = "successful" if pass_event["successful"] else "unsuccessful"
        results["pass_stats"]["pass_types"][pass_type]["total"] += 1
        results["pass_stats"]["pass_types"][pass_type][success_key] += 1
        
        if pass_event["successful"]:
            results["pass_stats"]["successful_passes"] += 1
        else:
            results["pass_stats"]["unsuccessful_passes"] += 1
        
        results["passes"].append(pass_entry)

    # Calculate averages and team stats
    for player_id, player_data in results["players"].items():
        if player_data["frames_tracked"] > 0:
            player_data["avg_speed"] = float(sum(player_data["speeds"]) / player_data["frames_tracked"])
            
            # Add to team stats
            team_key = f"team{player_data['team']}"
            results["team_stats"][team_key]["total_distance"] += player_data["total_distance"]
            results["team_stats"][team_key]["avg_speed"] += player_data["avg_speed"]

        # Remove the full speed list to save space
        player_data.pop("speeds")

    # Calculate team averages
    total_frames = len(team_ball_control)
    for team in ["team1", "team2"]:
        team_players = len([p for p in results["players"].values() if f"{team[-1]}" == str(p["team"])])
        if team_players > 0:
            results["team_stats"][team]["avg_speed"] = float(results["team_stats"][team]["avg_speed"] / team_players)
        results["team_stats"][team]["possession_percentage"] = float(
            results["ball_possession"][f"{team}_frames"] / total_frames * 100
        )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert all numpy types to native Python types
    results = convert_to_native(results)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    print("\n=============================================")
    print("Football Stats Analyzer with Advanced Metrics")
    print("=============================================\n")
    
    # Set batch size for processing frames in chunks
    BATCH_SIZE = 100  # Adjust based on your system's memory

    # Read video frames in batches
    video_path = 'input_videos/3.mp4'  # change source video here
    
    # Get video info first
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Total video frames: {total_frames}")
    
    # Initialize tracker outside of batch processing
    tracker = Tracker('models/best.pt')
    
    # Load tracking data from stub if available
    try:
        tracks = tracker.get_object_tracks(None, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
        using_stubs = True
    except:
        tracks = {'players': [], 'ball': [], 'referees': []}  # Initialize with all required keys
        using_stubs = False
    
    # Ensure 'referees' key exists in tracks
    if 'referees' not in tracks:
        tracks['referees'] = [{} for _ in range(len(tracks['players']))]
    
    # Process stubs first if available
    if using_stubs:
        num_track_frames = len(tracks['players'])
        print(f"Loaded {num_track_frames} frames from tracking stubs")
        
        # If using stubs, we still need to read video for visualization
        all_video_frames = []
        for start_idx in range(0, min(total_frames, num_track_frames), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_frames, num_track_frames)
            print(f"Reading video batch: frames {start_idx} to {end_idx}")
            
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            batch_frames = []
            
            for _ in range(end_idx - start_idx):
                ret, frame = cap.read()
                if ret:
                    batch_frames.append(frame)
                else:
                    break
            
            cap.release()
            all_video_frames.extend(batch_frames)
            
            # Force garbage collection
            del batch_frames
            gc.collect()
    else:
        # Process video in batches for tracking
        all_video_frames = []
        for start_idx in range(0, total_frames, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_frames)
            print(f"Processing batch: frames {start_idx} to {end_idx}")
            
            # Read batch of frames
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            batch_frames = []
            
            for _ in range(end_idx - start_idx):
                ret, frame = cap.read()
                if ret:
                    batch_frames.append(frame)
                else:
                    break
            
            cap.release()
            
            # Track objects in the batch
            batch_tracks = tracker.get_object_tracks(batch_frames, read_from_stub=False)
            
            # Merge batch tracking results with overall tracks
            tracks['players'].extend(batch_tracks['players'])
            tracks['ball'].extend(batch_tracks['ball'])
            
            # Ensure we also handle referees properly
            if 'referees' in batch_tracks:
                tracks['referees'].extend(batch_tracks['referees'])
            else:
                # If no referees detected, add empty dictionaries
                tracks['referees'].extend([{} for _ in range(len(batch_tracks['players']))])
            
            # Keep frames for visualization
            all_video_frames.extend(batch_frames)
            
            # Force garbage collection
            del batch_frames, batch_tracks
            gc.collect()
    
    # Ensure we only process frames that exist in both video and tracking data
    num_track_frames = len(tracks['players'])
    num_video_frames = len(all_video_frames)
    num_frames = min(num_track_frames, num_video_frames)
    
    print(f"Number of tracking frames: {num_track_frames}")
    print(f"Number of video frames: {num_video_frames}")
    print(f"Will process {num_frames} frames")

    # Trim tracks to match video length if necessary
    if num_track_frames > num_frames:
        for key in tracks:
            tracks[key] = tracks[key][:num_frames]

    print("\nAnalyzing player positions and movements...")
    # Process position data in parallel
    def add_positions_batch(start_idx, end_idx):
        tracker.add_position_to_tracks({
            'players': tracks['players'][start_idx:end_idx],
            'ball': tracks['ball'][start_idx:end_idx]
        })
    
    # Use thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for start_idx in range(0, num_frames, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_frames)
            futures.append(executor.submit(add_positions_batch, start_idx, end_idx))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()
    
    # Create view transformer
    view_transformer = ViewTransformer()
    
    # Process transformations in parallel
    def transform_positions_batch(start_idx, end_idx):
        view_transformer.add_transformed_position_to_tracks({
            'players': tracks['players'][start_idx:end_idx],
            'ball': tracks['ball'][start_idx:end_idx]
        })
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for start_idx in range(0, num_frames, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_frames)
            futures.append(executor.submit(transform_positions_batch, start_idx, end_idx))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()
    
    # Interpolate ball positions - this shouldn't be parallelized as it needs the entire sequence
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    print("Calculating player speeds and distances...")
    # Create speed estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    
    # Process speed calculations in parallel
    def calculate_speed_batch(start_idx, end_idx):
        speed_and_distance_estimator.add_speed_and_distance_to_tracks({
            'players': tracks['players'][start_idx:end_idx],
            'ball': tracks['ball'][start_idx:end_idx]
        })
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for start_idx in range(0, num_frames, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_frames)
            futures.append(executor.submit(calculate_speed_batch, start_idx, end_idx))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()
    
    print("Identifying team affiliations...")
    team_assigner = TeamAssigner()
    if tracks['players'] and tracks['players'][0]:  # Check if we have player tracks
        team_assigner.assign_team_color(all_video_frames[0], tracks['players'][0])
    else:
        print("Warning: No player tracks found in first frame")
        return
    
    # Process team assignments in parallel batches
    def assign_teams_batch(start_idx, end_idx):
        for frame_num in range(start_idx, end_idx):
            player_track = tracks['players'][frame_num]
            for player_id, track in player_track.items():
                if 'bbox' not in track:
                    continue
                team = team_assigner.get_player_team(all_video_frames[frame_num],
                                                  track['bbox'],
                                                  player_id)
                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for start_idx in range(0, num_frames, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_frames)
            futures.append(executor.submit(assign_teams_batch, start_idx, end_idx))
        
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            future.result()
    
    print("Analyzing ball possession...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    last_team = None
    
    # Ball possession analysis cannot be easily parallelized due to the dependency on last_team
    # But we can optimize the loop
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        has_valid_ball = (frame_num < len(tracks['ball']) and 
                          1 in tracks['ball'][frame_num] and 
                          'bbox' in tracks['ball'][frame_num][1])
        
        if not has_valid_ball:
            team_ball_control.append(last_team if last_team is not None else 1)
            continue
            
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1 and assigned_player in player_track and 'team' in player_track[assigned_player]:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            current_team = player_track[assigned_player]['team']
            team_ball_control.append(current_team)
            last_team = current_team
        else:
            team_ball_control.append(last_team if last_team is not None else 1)
    team_ball_control = np.array(team_ball_control)

    # Run detectors in parallel
    print("\nRunning multiple detectors in parallel...")
    
    shot_events = []
    dribble_events = []
    pass_events = []
    
    # Define thread functions for each detector
    def detect_shots():
        nonlocal shot_events
        print("Analyzing ball trajectories for shot attempts...")
        shot_detector = ShotDetector()
        shot_events = shot_detector.detect_goals(all_video_frames[:num_frames], tracks)
    
    def detect_dribbles():
        nonlocal dribble_events
        print("Analyzing player movements for dribbling actions...")
        dribble_detector = DribbleDetector()
        dribble_events = dribble_detector.detect_dribbles(all_video_frames[:num_frames], tracks, team_ball_control)
    
    def detect_passes():
        nonlocal pass_events
        print("Analyzing ball movements for passing events...")
        pass_detector = PassDetector()
        pass_events = pass_detector.detect_passes(all_video_frames[:num_frames], tracks, team_ball_control)
    
    # Create and start threads for detectors
    shot_thread = threading.Thread(target=detect_shots)
    dribble_thread = threading.Thread(target=detect_dribbles)
    pass_thread = threading.Thread(target=detect_passes)
    
    shot_thread.start()
    dribble_thread.start()
    pass_thread.start()
    
    # Wait for all detector threads to complete
    shot_thread.join()
    dribble_thread.join()
    pass_thread.join()

    # Print shot summary 
    print("\n======== SHOT DETECTION ========")
    if shot_events:
        print(f"\nDetected {len(shot_events)} shot attempts:")
        close_shots = len([s for s in shot_events if s["type"] == "close"])
        long_shots = len([s for s in shot_events if s["type"] == "long"])
        goals = len([s for s in shot_events if s["successful"]])
        
        print(f"  - Close shots: {close_shots}")
        print(f"  - Long shots: {long_shots}")
        print(f"  - Goals: {goals}")
        print(f"  - Conversion rate: {goals/len(shot_events)*100:.1f}%")
        
        # Print team stats
        team1_shots = len([s for s in shot_events if s.get("team") == 1])
        team2_shots = len([s for s in shot_events if s.get("team") == 2])
        team1_goals = len([s for s in shot_events if s.get("team") == 1 and s["successful"]])
        team2_goals = len([s for s in shot_events if s.get("team") == 2 and s["successful"]])
        
        print("\nTeam Shot Statistics:")
        print(f"  Team 1: {team1_shots} shots, {team1_goals} goals")
        print(f"  Team 2: {team2_shots} shots, {team2_goals} goals")
    else:
        print("No shots detected in this video segment.")
    print("===============================\n")
    
    # Print dribble summary
    print("\n====== DRIBBLE DETECTION ======")
    if dribble_events:
        print(f"\nDetected {len(dribble_events)} dribbling actions:")
        
        # Group dribbles by type
        regular_dribbles = len([d for d in dribble_events if d["type"] == "dribble" and not d["is_special"]])
        one_v_one = len([d for d in dribble_events if d["type"] == "one_vs_one" and not d["is_special"]])
        one_v_two = len([d for d in dribble_events if d["type"] == "one_vs_two" and not d["is_special"]])
        special_dribbles = len([d for d in dribble_events if d["is_special"]])
        
        successful_dribbles = len([d for d in dribble_events if d["successful"]])
        unsuccessful_dribbles = len([d for d in dribble_events if not d["successful"]])
        
        print(f"  - Regular dribbles (3+ opponents): {regular_dribbles}")
        print(f"  - 1v1 dribbles: {one_v_one}")
        print(f"  - 1v2 dribbles: {one_v_two}")
        print(f"  - Special dribbles: {special_dribbles}")
        print(f"  - Success rate: {successful_dribbles/len(dribble_events)*100:.1f}%")
        
        # Print team stats
        team1_dribbles = len([d for d in dribble_events if d.get("team") == 1])
        team2_dribbles = len([d for d in dribble_events if d.get("team") == 2])
        team1_success = len([d for d in dribble_events if d.get("team") == 1 and d["successful"]])
        team2_success = len([d for d in dribble_events if d.get("team") == 2 and d["successful"]])
        
        print("\nTeam Dribble Statistics:")
        print(f"  Team 1: {team1_dribbles} dribbles, {team1_success} successful ({team1_success/team1_dribbles*100:.1f}% success rate)" if team1_dribbles > 0 else "  Team 1: 0 dribbles")
        print(f"  Team 2: {team2_dribbles} dribbles, {team2_success} successful ({team2_success/team2_dribbles*100:.1f}% success rate)" if team2_dribbles > 0 else "  Team 2: 0 dribbles")
    else:
        print("No dribbling actions detected in this video segment.")
    print("===============================\n")
    
    # Print pass summary
    print("\n======== PASS DETECTION ========")
    if pass_events:
        completed_passes = len([p for p in pass_events if p["successful"] is True])
        failed_passes = len([p for p in pass_events if p["successful"] is False])
        success_rate = completed_passes / len(pass_events) * 100 if pass_events else 0
        
        print(f"\nDetected {len(pass_events)} passing events:")
        print(f"  - Successful passes: {completed_passes}")
        print(f"  - Unsuccessful passes: {failed_passes}")
        print(f"  - Pass completion rate: {success_rate:.1f}%")
        
        # Group by pass type
        long_ground = len([p for p in pass_events if p.get("pass_type") == "long_ground_pass"])
        long_aerial = len([p for p in pass_events if p.get("pass_type") == "long_aerial_pass"])
        short_ground = len([p for p in pass_events if p.get("pass_type") == "short_ground_pass"])
        short_aerial = len([p for p in pass_events if p.get("pass_type") == "short_aerial_pass"])
        
        print("\nPass Types:")
        print(f"  - Long ground passes: {long_ground}")
        print(f"  - Long aerial passes: {long_aerial}")
        print(f"  - Short ground passes: {short_ground}")
        print(f"  - Short aerial passes: {short_aerial}")
        
        # Team stats
        team1_passes = len([p for p in pass_events if p.get("sender_team") == 1])
        team2_passes = len([p for p in pass_events if p.get("sender_team") == 2])
        team1_completed = len([p for p in pass_events if p.get("sender_team") == 1 and p["successful"] is True])
        team2_completed = len([p for p in pass_events if p.get("sender_team") == 2 and p["successful"] is True])
        
        print("\nTeam Pass Statistics:")
        print(f"  Team 1: {team1_passes} passes, {team1_completed} completed" + 
              (f" ({team1_completed/team1_passes*100:.1f}% completion)" if team1_passes > 0 else ""))
        print(f"  Team 2: {team2_passes} passes, {team2_completed} completed" + 
              (f" ({team2_completed/team2_passes*100:.1f}% completion)" if team2_passes > 0 else ""))
    else:
        print("No passing events detected in this video segment.")
    print("===============================\n")
    
    # Save tracking results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'results/tracking_results_{timestamp}.json'
    save_tracking_results(tracks, team_ball_control, shot_events, dribble_events, pass_events, results_path)
    
    # Process output video in batches to reduce memory usage
    print("Generating annotated video with player tracking...")
    os.makedirs('output_videos', exist_ok=True)
    
    # Create video writer
    first_frame = all_video_frames[0]
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_videos/output_video.avi', fourcc, 30.0, (width, height))
    
    # Process and write frames in batches
    for start_idx in range(0, num_frames, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, num_frames)
        print(f"Processing output video batch: frames {start_idx} to {end_idx}")
        
        # Get batch frames
        batch_frames = all_video_frames[start_idx:end_idx]
        
        # Create batch tracks
        batch_tracks = {
            'players': tracks['players'][start_idx:end_idx],
            'ball': tracks['ball'][start_idx:end_idx],
            'referees': tracks['referees'][start_idx:end_idx]  # Add referees to the batch
        }
        
        # Get ball control for batch
        batch_ball_control = team_ball_control[start_idx:end_idx]
        
        # Draw annotations
        annotated_frames = tracker.draw_annotations(batch_frames, batch_tracks, batch_ball_control)
        speed_and_distance_estimator.draw_speed_and_distance(annotated_frames, batch_tracks)
        
        # Write frames
        for frame in annotated_frames:
            out.write(frame)
        
        # Force garbage collection
        del batch_frames, batch_tracks, annotated_frames
        gc.collect()
    
    # Release video writer
    out.release()
    
    print(f"Processing complete. Results saved to {results_path}")
    print("Video with player tracking saved to output_videos/output_video.avi")
    print(f"Shot, dribble, and pass statistics are available in the JSON file: {results_path}")

if __name__ == '__main__':
    main()