import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from typing import List, Tuple, Dict
import argparse
from pathlib import Path
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

os.environ['OPENBLAS_NUM_THREADS'] = '64'

class OpticalFlowAnalyzer:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
        self.model.eval()
        
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a frame for RAFT model."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame = frame.unsqueeze(0) / 255.0
        return frame.to(self.device)
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Compute optical flow between two consecutive frames."""
        with torch.no_grad():
            frame1_tensor = self.preprocess_frame(frame1)
            frame2_tensor = self.preprocess_frame(frame2)
            
            flow = self.model(frame1_tensor, frame2_tensor)[-1]
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            
        return flow
    
    def analyze_motion_regions(self, flow: np.ndarray, num_clusters: int = 3) -> Tuple[np.ndarray, Dict]:
        """Cluster motion regions based on optical flow magnitude and direction."""
        h, w = flow.shape[:2]
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        direction = np.arctan2(flow[..., 1], flow[..., 0])
        
        # Create feature matrix for clustering
        features = np.zeros((h * w, 3))
        features[:, 0] = magnitude.ravel()
        features[:, 1] = np.cos(direction).ravel()
        features[:, 2] = np.sin(direction).ravel()
        
        # Normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42,)
        labels = kmeans.fit_predict(features)
        labels = labels.reshape(h, w)
        
        # Analyze clusters
        cluster_stats = {}
        for i in range(num_clusters):
            cluster_mask = (labels == i)
            cluster_magnitude = magnitude[cluster_mask]
            cluster_stats[i] = {
                'mean_magnitude': np.mean(cluster_magnitude),
                'std_magnitude': np.std(cluster_magnitude),
                'pixel_count': np.sum(cluster_mask),
                'is_static': np.mean(cluster_magnitude) < 0.1  # Threshold for static regions
            }
        
        return labels, cluster_stats
    
    def process_video(self, video_path: str, output_path: str = None) -> List[Tuple[np.ndarray, Dict]]:
        """Process a video and return motion analysis results for each frame pair."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        results = []
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames-1, desc="Processing video")
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
                
            flow = self.compute_optical_flow(prev_frame, curr_frame)
            labels, stats = self.analyze_motion_regions(flow)
            
            if output_path:
                # Visualize results
                vis_frame = curr_frame.copy()
                for i, stat in stats.items():
                    if not stat['is_static']:
                        mask = (labels == i).astype(np.uint8) * 255
                        print("mask:",mask.shape)
                        print("vis_frame:",vis_frame.shape)
                        mask = np.expand_dims(mask, axis=-1).repeat(3, axis=-1)
                        print("mask:",mask.shape)
                        
                        vis_frame[mask > 0] = cv2.addWeighted(vis_frame[mask > 0], 0.7, 255, 0.3, 0)
                
                cv2.imwrite(f"{output_path}/frame_{len(results):04d}.jpg", vis_frame)
            
            results.append((labels, stats))
            prev_frame = curr_frame
            pbar.update(1)
        
        cap.release()
        pbar.close()
        return results

def main():
    parser = argparse.ArgumentParser(description='Analyze motion regions in a video using RAFT optical flow')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to output directory for visualization')
    parser.add_argument('--clusters', type=int, default=3, help='Number of motion clusters')
    args = parser.parse_args()
    
    analyzer = OpticalFlowAnalyzer()
    results = analyzer.process_video(args.video, args.output)
    
    # Print summary statistics
    print("\nMotion Analysis Summary:")
    for i, (_, stats) in enumerate(results):
        print(f"\nFrame {i+1}:")
        for cluster_id, stat in stats.items():
            motion_type = "Static" if stat['is_static'] else "Moving"
            print(f"  Cluster {cluster_id} ({motion_type}):")
            print(f"    Mean magnitude: {stat['mean_magnitude']:.4f}")
            print(f"    Pixel count: {stat['pixel_count']}")

if __name__ == "__main__":
    main()

