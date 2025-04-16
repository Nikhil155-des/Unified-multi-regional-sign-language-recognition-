import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature extractor: ResNet18
model = resnet18(pretrained=True)
model.fc = nn.Identity()
model = model.to(device).eval()

# Image preprocessing
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def extract_video_embedding(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame).unsqueeze(0))
    cap.release()
    if not frames:
        return torch.zeros(512).to(device)
    batch = torch.cat(frames).to(device)
    with torch.no_grad():
        features = model(batch)  # shape: [n_frames, 512]
    return features.mean(dim=0)  # mean feature of all frames

def get_word_embeddings(root_dir):
    word_embeddings = {}
    for word in tqdm(os.listdir(root_dir), desc=f"Processing {root_dir}"):
        word_path = os.path.join(root_dir, word)
        if not os.path.isdir(word_path):
            continue
        video_features = []
        for video_file in os.listdir(word_path):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(word_path, video_file)
                embedding = extract_video_embedding(video_path)
                video_features.append(embedding)
        if video_features:
            word_embeddings[word] = torch.stack(video_features).mean(dim=0).cpu().numpy()
    return word_embeddings

# Paths
# Use raw strings (r'') and avoid ending the path with a backslash
bsl_nzsl_path = "D:\\Engeneering\\Third Year\\SEM6\\Unified-multi-regional-sign-language-recognition-\\dataset\\train\\BSL_NZSL"
isl_auslan_path = "D:\\Engeneering\\Third Year\\SEM6\\Unified-multi-regional-sign-language-recognition-\\dataset\\test\\ISL_Auslan"


# Extract embeddings
embeddings_bsl_nzsl = get_word_embeddings(bsl_nzsl_path)
embeddings_isl_auslan = get_word_embeddings(isl_auslan_path)

# Similarity computation
print("\n=== Cosine Similarity Between Words Across Languages ===")
for word in embeddings_bsl_nzsl:
    if word in embeddings_isl_auslan:
        sim = cosine_similarity(
            embeddings_bsl_nzsl[word].reshape(1, -1),
            embeddings_isl_auslan[word].reshape(1, -1)
        )[0][0]
        print(f"Word: {word:15} | Similarity: {sim:.4f}")
    else:
        print(f"Word: {word:15} | No matching word found in ISL_Auslan")
