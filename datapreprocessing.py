def extract_keypoints(frame):
    """Extract keypoints using MediaPipe from a frame"""
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    
    # Initialize keypoints
    keypoints = []
    
    # Extract pose keypoints (33 landmarks * 2 coordinates)
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y])
    else:
        keypoints.extend([0] * (33 * 2))
    
    # Extract hand keypoints (21 landmarks * 2 hands * 2 coordinates)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y])
            # If only one hand is detected, pad for the second hand
            if len(hands_results.multi_hand_landmarks) == 1:
                keypoints.extend([0] * (21 * 2))
    else:
        keypoints.extend([0] * (21 * 2 * 2))
    
    # Extract face keypoints (using a subset of 50 important facial landmarks * 2 coordinates)
    if face_results.multi_face_landmarks:
        # Select important facial landmarks (e.g., eyes, eyebrows, mouth)
        important_landmarks = [0, 17, 61, 78, 80, 91, 95, 146, 152, 234, 246, 331, 397, 454]
        important_landmarks.extend(range(61, 69))  # Left eye
        important_landmarks.extend(range(291, 299))  # Right eye
        important_landmarks.extend(range(0, 17))  # Jawline
        important_landmarks.extend(range(46, 55))  # Eyebrows
        
        # Keep only unique landmarks and limit to 50
        important_landmarks = list(set(important_landmarks))[:50]
        
        for idx in important_landmarks:
            if idx < len(face_results.multi_face_landmarks[0].landmark):
                landmark = face_results.multi_face_landmarks[0].landmark[idx]
                keypoints.extend([landmark.x, landmark.y])
    else:
        keypoints.extend([0] * (50 * 2))
    
    return np.array(keypoints, dtype=np.float32)

def preprocess_video(video_path, target_frames=30):
    """Load video, extract frames, and preprocess for the model"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    keypoints_seq = []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure we can get enough frames
    if total_frames == 0:
        print(f"Warning: Could not read frames from {video_path}")
        return None, None
    
    # Calculate sampling rate to get target_frames
    sampling_rate = max(1, total_frames // target_frames)
    
    frame_indices = list(range(0, min(total_frames, sampling_rate * target_frames), sampling_rate))
    if len(frame_indices) > target_frames:
        frame_indices = frame_indices[:target_frames]
    
    # If we don't have enough frames, repeat the last frame
    while len(frame_indices) < target_frames:
        frame_indices.append(frame_indices[-1] if frame_indices else 0)
    
    # Read frames at specific indices
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        frame = cv2.resize(frame, (CONFIG['input_shape'][0], CONFIG['input_shape'][1]))
        
        # Normalize pixel values
        normalized_frame = frame.astype(np.float32) / 255.0
        
        # Extract keypoints
        keypoints = extract_keypoints(frame)
        
        frames.append(normalized_frame)
        keypoints_seq.append(keypoints)
    
    cap.release()
    
    # If we couldn't extract enough frames, return None
    if len(frames) < target_frames:
        print(f"Warning: Only got {len(frames)} frames from {video_path}")
        return None, None
    
    return np.array(frames), np.array(keypoints_seq)

def load_dataset(base_path, language_dirs, split='train'):
    """Load dataset from the given directory structure"""
    videos = []
    keypoints = []
    labels = []
    label_to_idx = {}
    language_ids = []
    
    label_idx = 0
    for lang_idx, lang_dir in enumerate(language_dirs):
        full_path = os.path.join(base_path, split, lang_dir)
        if not os.path.exists(full_path):
            print(f"Warning: Path {full_path} does not exist")
            continue
            
        for word_dir in os.listdir(full_path):
            word_path = os.path.join(full_path, word_dir)
            if not os.path.isdir(word_path):
                continue
                
            # Assign label index if new
            if word_dir not in label_to_idx:
                label_to_idx[word_dir] = label_idx
                label_idx += 1
                
            for video_file in os.listdir(word_path):
                if not video_file.endswith('.mp4'):
                    continue
                    
                video_path = os.path.join(word_path, video_file)
                frames, kp_seq = preprocess_video(video_path)
                
                if frames is not None and kp_seq is not None:
                    videos.append(frames)
                    keypoints.append(kp_seq)
                    labels.append(label_to_idx[word_dir])
                    language_ids.append(lang_idx)
    
    # Convert to arrays
    videos = np.array(videos)
    keypoints = np.array(keypoints)
    labels = np.array(labels)
    language_ids = np.array(language_ids)
    
    # Create inverse mapping
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    return videos, keypoints, labels, language_ids, label_to_idx, idx_to_label

# Data Augmentation Functions
def augment_video(video, keypoints):
    """Apply augmentation to video frames and keypoints"""
    augmented_video = video.copy()
    augmented_keypoints = keypoints.copy()
    
    # Randomly select augmentation type
    aug_type = np.random.choice(['flip', 'rotate', 'zoom', 'brightness', 'none'])
    
    if aug_type == 'flip' and np.random.random() < 0.5:
        # Horizontal flip
        augmented_video = augmented_video[:, :, ::-1, :]
        
        # Flip keypoints (x coordinates)
        for i in range(len(augmented_keypoints)):
            # Extract x coordinates (even indices)
            x_coords = augmented_keypoints[i, ::2]
            # Flip x coordinates (1.0 - x)
            augmented_keypoints[i, ::2] = 1.0 - x_coords
    
    elif aug_type == 'rotate' and np.random.random() < 0.5:
        # Random rotation (small angle)
        angle = np.random.uniform(-15, 15)
        
        for i in range(len(augmented_video)):
            frame = augmented_video[i]
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_video[i] = cv2.warpAffine(frame, rotation_matrix, (width, height))
            
            # Transform keypoints accordingly
            keypoints_frame = augmented_keypoints[i].reshape(-1, 2)
            ones = np.ones(shape=(len(keypoints_frame), 1))
            keypoints_ones = np.hstack([keypoints_frame, ones])
            transformed = keypoints_ones.dot(rotation_matrix.T)
            augmented_keypoints[i] = transformed.reshape(-1)
    
    elif aug_type == 'zoom' and np.random.random() < 0.5:
        # Random zoom
        zoom_factor = np.random.uniform(0.8, 1.2)
        
        for i in range(len(augmented_video)):
            frame = augmented_video[i]
            height, width = frame.shape[:2]
            
            # Calculate new dimensions
            new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
            
            # Resize frame
            resized = cv2.resize(frame, (new_width, new_height))
            
            # Crop or pad to maintain original size
            if zoom_factor > 1:  # Zoomed in, need to crop
                start_y = (new_height - height) // 2
                start_x = (new_width - width) // 2
                augmented_video[i] = resized[start_y:start_y+height, start_x:start_x+width]
            else:  # Zoomed out, need to pad
                start_y = (height - new_height) // 2
                start_x = (width - new_width) // 2
                augmented_video[i] = np.zeros_like(frame)
                augmented_video[i][start_y:start_y+new_height, start_x:start_x+new_width] = resized
                
            # Adjust keypoints
            keypoints_frame = augmented_keypoints[i].reshape(-1, 2)
            keypoints_frame = keypoints_frame * zoom_factor
            
            if zoom_factor > 1:  # Zoomed in
                keypoints_frame[:, 0] -= (new_width - width) / (2 * new_width)
                keypoints_frame[:, 1] -= (new_height - height) / (2 * new_height)
            else:  # Zoomed out
                keypoints_frame[:, 0] += (width - new_width) / (2 * width)
                keypoints_frame[:, 1] += (height - new_height) / (2 * height)
                
            augmented_keypoints[i] = keypoints_frame.reshape(-1)
    
    elif aug_type == 'brightness' and np.random.random() < 0.5:
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        augmented_video = augmented_video * brightness
        augmented_video = np.clip(augmented_video, 0, 1.0)
    
    return augmented_video, augmented_keypoints
