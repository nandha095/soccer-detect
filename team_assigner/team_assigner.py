from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self,image):
        image_2d = image.reshape(-1,3)

        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        # Ensure bbox coordinates are within frame boundaries
        height, width = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(width, int(bbox[2]))
        y2 = min(height, int(bbox[3]))
        
        # Check if bbox is valid
        if x2 <= x1 or y2 <= y1:
            print(f"Invalid bbox dimensions: {bbox}")
            return None
            
        image = frame[y1:y2, x1:x2]
        
        # Check if extracted image is valid
        if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            print(f"Empty image extracted from bbox: {bbox}")
            return None
            
        top_half_image = image[0:max(1, int(image.shape[0]/2)), :]
        
        # Ensure we have enough pixels for clustering
        if top_half_image.size == 0 or top_half_image.shape[0] * top_half_image.shape[1] < 4:
            print(f"Not enough pixels for clustering in top half: {top_half_image.shape}")
            return None

        kmeans = self.get_clustering_model(top_half_image)
        
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], 
                         clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color


    def assign_team_color(self,frame, player_detections):
        player_colors = []
        valid_colors = []
        
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            if player_color is not None:
                valid_colors.append(player_color)
        
        if len(valid_colors) < 2:
            raise ValueError("Not enough valid player detections to determine team colors")
            
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(valid_colors)
        
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
            
        player_color = self.get_player_color(frame, player_bbox)
        if player_color is None:
            # If we can't determine the color, assign to team 1 as default
            team_id = 1
        else:
            team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] + 1
            
        self.player_team_dict[player_id] = team_id
        return team_id