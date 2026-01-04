"""
Image Retrieval System with Relevance Feedback
CS419 - Assignment 02
Using CLIP/CNN Features + Color Histogram with Rocchio Algorithm
"""

import numpy as np
import os
from PIL import Image
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2

# Deep learning imports
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Try to import CLIP for better semantic understanding
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: open-clip-torch not installed. Using ResNet50 only.")


class FeatureExtractor:
    """Extract features from images using CLIP/CNN and Color Histogram"""

    def __init__(self, use_gpu: bool = True, model_type: str = "clip"):
        """
        Initialize feature extractor.

        Args:
            use_gpu: Whether to use GPU if available
            model_type: "clip" (recommended for anime) or "resnet50"
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_type = model_type if (model_type == "resnet50" or not CLIP_AVAILABLE) else "clip"
        print(f"Using device: {self.device}")
        print(f"Using model: {self.model_type.upper()}")

        # Color histogram dimension (must be set before model init)
        self.color_dim = 512  # 8*8*8 bins

        if self.model_type == "clip" and CLIP_AVAILABLE:
            self._init_clip()
        else:
            self._init_resnet()

    def _init_clip(self):
        """Initialize CLIP model - much better for anime/semantic understanding"""
        # Use ViT-B-32 for good balance of speed and quality
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.cnn_dim = 512  # CLIP ViT-B-32 dimension
        self.total_dim = self.cnn_dim + self.color_dim
        print("CLIP model loaded (ViT-B-32) - Better for anime!")

    def _init_resnet(self):
        """Initialize ResNet50 model"""
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.cnn_dim = 2048  # ResNet50 dimension
        self.total_dim = self.cnn_dim + self.color_dim
        print("ResNet50 model loaded")

    def extract_color_histogram(self, image: Image.Image) -> np.ndarray:
        """Extract color histogram in HSV space"""
        img_array = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)

        return hist.astype(np.float32)

    def extract_cnn_features(self, image: Image.Image) -> np.ndarray:
        """Extract deep features using CLIP or ResNet50"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.model_type == "clip" and CLIP_AVAILABLE:
                features = self.model.encode_image(img_tensor)
                features = features.squeeze().cpu().numpy()
            else:
                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()

        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-7)
        return features.astype(np.float32)

    def extract_features(self, image: Image.Image, cnn_weight: float = 0.85) -> np.ndarray:
        """
        Extract combined CNN + Color histogram features.

        Args:
            image: PIL Image
            cnn_weight: Weight for CNN features (0-1). Higher = more semantic.
                       Default 0.85 for CLIP (semantic-focused)
        """
        cnn_feat = self.extract_cnn_features(image)
        color_feat = self.extract_color_histogram(image)

        color_weight = 1.0 - cnn_weight
        combined = np.concatenate([cnn_feat * cnn_weight, color_feat * color_weight])

        return combined


class ImageDatabase:
    """Manage image database with pre-computed features"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.extractor = feature_extractor
        self.features: np.ndarray = None  # N x D feature matrix
        self.image_paths: List[str] = []
        self.index_to_path: Dict[int, str] = {}
    
    def build_database(self, image_folder: str, cache_file: str = None,
                        progress_callback=None):
        """Build database from image folder

        Args:
            image_folder: Path to folder containing images
            cache_file: Name of cache file to save/load features (auto-generated if None)
            progress_callback: Optional callback function(current, total, message) for progress updates
        """
        import time

        image_folder = Path(image_folder)

        # Auto-generate cache filename based on model type
        if cache_file is None:
            model_type = getattr(self.extractor, 'model_type', 'unknown')
            cache_file = f"features_cache_{model_type}.pkl"

        def format_time(seconds):
            """Format seconds into human readable string"""
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                mins = seconds // 60
                secs = seconds % 60
                return f"{mins:.0f}m {secs:.0f}s"
            else:
                hours = seconds // 3600
                mins = (seconds % 3600) // 60
                return f"{hours:.0f}h {mins:.0f}m"

        def update_progress(current, total, message):
            if progress_callback:
                progress_callback(current, total, message)
            if current % 100 == 0 or current == total:
                print(message)

        # Check for cached features
        cache_path = image_folder / cache_file
        if cache_path.exists():
            update_progress(0, 1, f"Loading cached features from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
                self.features = cache['features']
                self.image_paths = cache['image_paths']
                self.index_to_path = {i: p for i, p in enumerate(self.image_paths)}
                update_progress(1, 1, f"Loaded {len(self.image_paths)} images from cache")
                return

        # Find all images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif']
        image_files = []
        for ext in extensions:
            image_files.extend(image_folder.rglob(ext))

        total_images = len(image_files)
        update_progress(0, total_images, f"Found {total_images} images to process")

        # Extract features with timing
        features_list = []
        valid_paths = []
        start_time = time.time()
        last_update_time = start_time

        for i, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path)
                feat = self.extractor.extract_features(img)
                features_list.append(feat)
                valid_paths.append(str(img_path))

                # Calculate speed and ETA every 10 images or at least every 2 seconds
                current_time = time.time()
                if (i + 1) % 10 == 0 or (current_time - last_update_time) >= 2 or (i + 1) == total_images:
                    elapsed = current_time - start_time
                    images_done = i + 1
                    images_per_sec = images_done / elapsed if elapsed > 0 else 0
                    remaining_images = total_images - images_done
                    eta_seconds = remaining_images / images_per_sec if images_per_sec > 0 else 0

                    pct = images_done * 100 // total_images
                    speed_str = f"{images_per_sec:.1f} img/s"
                    eta_str = format_time(eta_seconds) if remaining_images > 0 else "Done!"

                    message = f"Processed {images_done}/{total_images} ({pct}%) | Speed: {speed_str} | ETA: {eta_str}"
                    update_progress(images_done, total_images, message)
                    last_update_time = current_time

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        self.features = np.array(features_list)
        self.image_paths = valid_paths
        self.index_to_path = {i: p for i, p in enumerate(self.image_paths)}

        # Cache features
        update_progress(total_images, total_images, f"Saving features to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({'features': self.features, 'image_paths': self.image_paths}, f)

        update_progress(total_images, total_images, f"Database built with {len(self.image_paths)} images")
    
    def get_image(self, index: int) -> Image.Image:
        """Get image by index"""
        return Image.open(self.image_paths[index])
    
    def get_feature(self, index: int) -> np.ndarray:
        """Get feature vector by index"""
        return self.features[index]


class RelevanceFeedbackRetrieval:
    """Image retrieval with Rocchio relevance feedback"""
    
    def __init__(self, database: ImageDatabase, feature_extractor: FeatureExtractor):
        self.db = database
        self.extractor = feature_extractor
        
        # Rocchio parameters (from CS419 w8 slide)
        self.alpha = 1.0    # Weight for original query
        self.beta = 0.75    # Weight for relevant documents
        self.gamma = 0.25   # Weight for non-relevant documents
        
        # Current state
        self.current_query: np.ndarray = None
        self.iteration: int = 0
        self.query_history: List[np.ndarray] = []
    
    def initial_search(self, query_image: Image.Image, top_k: int = 20) -> List[Tuple[int, float]]:
        """Perform initial search with query image"""
        # Extract query features
        self.current_query = self.extractor.extract_features(query_image)
        self.iteration = 0
        self.query_history = [self.current_query.copy()]
        
        return self._search(top_k)
    
    def _search(self, top_k: int) -> List[Tuple[int, float]]:
        """Search database using current query vector"""
        # Compute cosine similarity
        query_norm = self.current_query / (np.linalg.norm(self.current_query) + 1e-7)
        db_norms = self.db.features / (np.linalg.norm(self.db.features, axis=1, keepdims=True) + 1e-7)
        
        similarities = np.dot(db_norms, query_norm)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def relevance_feedback(self, relevant_indices: List[int], 
                           non_relevant_indices: List[int],
                           top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Apply Rocchio relevance feedback algorithm
        
        Standard Rocchio Formula:
        q_new = α * q_old + β * (1/|Dr|) * Σ(d in Dr) - γ * (1/|Dnr|) * Σ(d in Dnr)
        
        Where:
        - q_old: original query vector
        - Dr: set of relevant documents
        - Dnr: set of non-relevant documents
        - α, β, γ: tunable weights
        """
        self.iteration += 1
        
        # Start with weighted original query
        new_query = self.alpha * self.current_query
        
        # Add centroid of relevant documents
        if len(relevant_indices) > 0:
            relevant_features = self.db.features[relevant_indices]
            relevant_centroid = np.mean(relevant_features, axis=0)
            new_query += self.beta * relevant_centroid
        
        # Subtract centroid of non-relevant documents
        if len(non_relevant_indices) > 0:
            non_relevant_features = self.db.features[non_relevant_indices]
            non_relevant_centroid = np.mean(non_relevant_features, axis=0)
            new_query -= self.gamma * non_relevant_centroid
        
        # Update query
        self.current_query = new_query
        self.query_history.append(self.current_query.copy())
        
        return self._search(top_k)
    
    def set_rocchio_params(self, alpha: float, beta: float, gamma: float):
        """Update Rocchio parameters"""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def reset(self):
        """Reset retrieval state"""
        self.current_query = None
        self.iteration = 0
        self.query_history = []


def compute_precision_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Compute Precision@K metric"""
    retrieved_at_k = retrieved[:k]
    relevant_count = sum(1 for idx in retrieved_at_k if idx in relevant)
    return relevant_count / k if k > 0 else 0.0


if __name__ == "__main__":
    # Test the system
    print("Image Retrieval System with Relevance Feedback")
    print("=" * 50)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(use_gpu=True)
    
    # Create database
    db = ImageDatabase(extractor)
    
    print("\nTo use, call:")
    print("  db.build_database('path/to/anime/images')")
    print("  retrieval = RelevanceFeedbackRetrieval(db, extractor)")
    print("  results = retrieval.initial_search(query_image)")
