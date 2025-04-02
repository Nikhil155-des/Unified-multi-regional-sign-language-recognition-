# Main training and evaluation function
def main():
    # Load the dataset
    print("Loading dataset...")
    train_videos, train_keypoints, train_labels, train_languages, train_label_to_idx, train_idx_to_label = load_dataset(
        CONFIG['dataset_path'], ['BSL_NZSL'], 'train'
    )
    
    test_videos, test_keypoints, test_labels, test_languages, test_label_to_idx, test_idx_to_label = load_dataset(
        CONFIG['dataset_path'], ['ISL_Auslan'], 'test'
    )
    
    # Update config with actual number of classes
    CONFIG['num_classes'] = len(train_label_to_idx)
    
    # Data augmentation for training
    print("Augmenting training data...")
    augmented_videos = []
    augmented_keypoints = []
    augmented_labels = []
    augmented_languages = []
    
    for i in tqdm(range(len(train_videos))):
        # Add original
        augmented_videos.append(train_videos[i])
        augmented_keypoints.append(train_keypoints[i])
        augmented_labels.append(train_labels[i])
        augmented_languages.append(train_languages[i])
        
        # Add augmented
        aug_video, aug_kp = augment_video(train_videos[i], train_keypoints[i])
        augmented_videos.append(aug_video)
        augmented_keypoints.append(aug_kp)
        augmented_labels.append(train_labels[i])
        augmented_languages.append(train_languages[i])
    
    train_videos = np.array(augmented_videos)
    train_keypoints = np.array(augmented_keypoints)
    train_labels = np.array(augmented_labels)
    train_languages = np.array(augmented_languages)
    
    # Create train/val split
    indices = np.arange(len(train_videos))
    np.random.shuffle(indices)
