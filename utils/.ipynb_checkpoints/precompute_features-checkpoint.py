"""
Utility script to precompute and save feature vectors for the gallery images.
This is useful for large datasets where computing features on-the-fly would be slow.

The script selects a specified number of random images per category (default: 10)
and creates a FAISS index and feature array for fast similarity search.

Usage:
    python precompute_features.py --model weights/model.pth --data train_val.json --output data/features.npy --faiss data/faiss_index.bin --num-per-class 10
"""

import argparse
import torch
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import random

# Add the parent directory to the path so we can import our modules
sys.path.append(".")
from src.model import ResNetTransferModel
from utils.image_utils import preprocess_image, extract_features
from utils.faiss_utils import build_faiss_index, save_faiss_index

def main():
    parser = argparse.ArgumentParser(description='Precompute features for gallery images')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--data', type=str, required=True, help='Path to the gallery data JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the features')
    parser.add_argument('--faiss', type=str, help='Path to save the FAISS index')
    parser.add_argument('--num-per-class', type=int, default=10, help='Number of images per class to use (default: 10)')
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ResNetTransferModel(num_classes=101, embedding_size=128, pretrained=False).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    # Load gallery data
    with open(args.data, 'r') as f:
        gallery_data = json.load(f)
    
    # Verify the structure
    if 'train' not in gallery_data or 'categories' not in gallery_data:
        print("Error: The data file must have 'train' and 'categories' keys")
        return
    
    categories = gallery_data['categories']
    print(f"Found {len(categories)} categories")
    
    # Group images by category
    images_by_category = {}
    for i, (label, img_path) in enumerate(gallery_data['train']):
        if label not in images_by_category:
            images_by_category[label] = []
        images_by_category[label].append((i, img_path))
    
    # Select random images from each category
    selected_images = []
    random.seed(42)  # For reproducibility
    
    for label, images in images_by_category.items():
        # Determine how many to select
        num_to_select = min(args.num_per_class, len(images))
        
        # Randomly select
        if len(images) > num_to_select:
            selected = random.sample(images, num_to_select)
        else:
            selected = images
        
        selected_images.extend(selected)
        
        # Get category name for display
        category_name = categories[label] if label < len(categories) else f"Category {label}"
        print(f"Category {category_name}: Selected {num_to_select} of {len(images)} images")
    
    print(f"\nProcessing {len(selected_images)} images for FAISS index...")
    
    # Extract features from selected images
    all_features = []
    all_paths = []
    
    for i, (orig_idx, img_path) in enumerate(tqdm(selected_images)):
        try:
            # Fix path if needed (relative paths in caltech101 folder)
            full_path = img_path
            if not os.path.exists(full_path):
                # Try with caltech101 prefix
                caltech_path = os.path.join("caltech101", img_path)
                if os.path.exists(caltech_path):
                    full_path = caltech_path
                else:
                    print(f"Warning: Could not find image at {img_path} or {caltech_path}")
                    continue
            
            # Load and preprocess the image
            image = Image.open(full_path).convert('RGB')
            image_tensor = preprocess_image(image, device)
            
            # Extract features
            features = extract_features(model, image_tensor, device)
            
            # Store features and path
            all_features.append(features)
            all_paths.append(full_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Stack all features
    if all_features:
        all_features = np.vstack(all_features)
        print(f"Extracted features shape: {all_features.shape}")
        
        # Save features as numpy array (for FAISS)
        np.save(args.output, all_features)
        print(f"Features saved to {args.output}")
        
        # Save paths array
        paths_file = os.path.splitext(args.output)[0] + "_paths.json"
        with open(paths_file, 'w') as f:
            json.dump(all_paths, f)
        print(f"Image paths saved to {paths_file}")
        
        # Create and save FAISS index if requested
        if args.faiss:
            index = build_faiss_index(all_features)
            save_faiss_index(index, args.faiss)
            print(f"FAISS index saved to {args.faiss}")
    else:
        print("No features extracted. Check that the image paths in the data file are correct.")

if __name__ == "__main__":
    main()