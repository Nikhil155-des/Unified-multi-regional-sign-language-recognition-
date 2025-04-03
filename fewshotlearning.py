# Define the Few-Shot Learning Component using Prototypical Networks
class PrototypicalNetworks:
    def __init__(self, model, support_set_size=5):
        self.model = model  # The main model that extracts features
        self.support_set_size = support_set_size
        self.prototypes = {}  # Dictionary mapping class label to prototype vector
        
    def create_prototypes(self, videos, keypoints, labels, language_ids):
        """Create prototypes for each class from the support set"""
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get indices of examples with this label
            indices = np.where(labels == label)[0]
            
            # Randomly select support_set_size examples (or less if not available)
            support_size = min(self.support_set_size, len(indices))
            support_indices = np.random.choice(indices, support_size, replace=False)
            
            # Extract features from support set examples
            support_features = []
            for idx in support_indices:
                # Process through the main model to get features
                visual_features, motion_features = self.model.get_features(
                    np.expand_dims(videos[idx], axis=0),
                    np.expand_dims(keypoints[idx], axis=0),
                    np.expand_dims(language_ids[idx], axis=0)
                )
                support_features.append(visual_features[0])  # Extract the feature vector
            
            # Compute prototype as mean of support features
            prototype = np.mean(support_features, axis=0)
            self.prototypes[label] = prototype
            
    def predict(self, video, keypoints, language_id):
        """Predict class for a new example using prototypes"""
        # Get features for the example
        visual_features, motion_features = self.model.get_features(
            np.expand_dims(video, axis=0),
            np.expand_dims(keypoints, axis=0),
            np.expand_dims(language_id, axis=0)
        )
        
        # Calculate distance to each prototype
        distances = {}
        for label, prototype in self.prototypes.items():
            # Use Euclidean distance
            distance = np.linalg.norm(visual_features[0] - prototype)
            distances[label] = distance
            
        # Return label with minimum distance
        return min(distances, key=distances.get)
