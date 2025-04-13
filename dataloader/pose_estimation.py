mp_pose = mp.solutions.holistic.Holistic(static_image_mode=False)

def extract_keypoints(frames):
    keypoints = []
    for frame in frames:
        results = mp_pose.process(frame)
        pose = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                pose.extend([lm.x, lm.y])
        keypoints.append(pose)
    return keypoints  # shape: [n_frames, 468*2]
