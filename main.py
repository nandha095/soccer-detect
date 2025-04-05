from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from transformation.transformer import ViewTransformer
from speed_and_distance import SpeedAndDistance_Estimator
import json
import os
from datetime import datetime

def save_tracking_results(tracks, team_ball_control, output_path):
    """Save tracking results to a JSON file."""
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "players": {},
        "team_stats": {
            "team1": {"total_distance": 0, "avg_speed": 0, "possession_time": 0},
            "team2": {"total_distance": 0, "avg_speed": 0, "possession_time": 0}
        },
        "ball_possession": {
            "team1_frames": int(np.sum(team_ball_control == 1)),
            "team2_frames": int(np.sum(team_ball_control == 2)),
        }
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
                    "ball_possession_frames": 0
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

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    video_frames = read_video('input_videos/2.mp4') ## change source video here
    print(f"Number of video frames: {len(video_frames)}")
    
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Ensure we only process frames that exist in both video and tracking data
    num_track_frames = len(tracks['players'])
    num_video_frames = len(video_frames)
    num_frames = min(num_track_frames, num_video_frames)
    
    print(f"Number of tracking frames: {num_track_frames}")
    print(f"Will process {num_frames} frames")

    # Trim tracks to match video length if necessary
    if num_track_frames > num_frames:
        for key in tracks:
            tracks[key] = tracks[key][:num_frames]

    tracker.add_position_to_tracks(tracks)
    
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    team_assigner = TeamAssigner()
    if tracks['players'] and tracks['players'][0]:  # Check if we have player tracks
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    else:
        print("Warning: No player tracks found in first frame")
        return
    
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        for player_id, track in player_track.items():
            if 'bbox' not in track:
                continue
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                track['bbox'],
                                                player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    last_team = None
    
    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        if (frame_num >= len(tracks['ball']) or 
            1 not in tracks['ball'][frame_num] or 
            'bbox' not in tracks['ball'][frame_num][1]):
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

    # Save tracking results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'results/tracking_results_{timestamp}.json'
    save_tracking_results(tracks, team_ball_control, results_path)
    
    # Process only available frames for output
    output_video_frames = tracker.draw_annotations(video_frames[:num_frames], 
                                                 tracks, 
                                                 team_ball_control)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)
    
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print(f"Processing complete. Results saved to {results_path}")

if __name__ == '__main__':
    main()