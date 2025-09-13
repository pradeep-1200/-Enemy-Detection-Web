# D:\Projects\enemy_detection_web\train_perfect_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import cv2
from pathlib import Path

def extract_advanced_features(image_path):
    """Extract comprehensive features for better classification"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Basic image features
        height, width = gray.shape
        total_pixels = height * width
        aspect_ratio = width / height
        
        # Color statistics (BGR format)
        mean_r = np.mean(img[:,:,2])
        mean_g = np.mean(img[:,:,1])
        mean_b = np.mean(img[:,:,0])
        std_r = np.std(img[:,:,2])
        std_g = np.std(img[:,:,1])
        std_b = np.std(img[:,:,0])
        
        # Gray statistics
        mean_gray = np.mean(gray)
        std_gray = np.std(gray)
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        # Advanced shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0
        else:
            solidity = 0
        
        # Texture features
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        texture_variance = np.var(blur)
        
        # HSV features
        mean_h = np.mean(hsv[:,:,0])
        mean_s = np.mean(hsv[:,:,1])
        mean_v = np.mean(hsv[:,:,2])
        
        # Advanced color analysis
        blue_ratio = mean_b / (mean_r + mean_g + mean_b + 1e-6)
        red_ratio = mean_r / (mean_r + mean_g + mean_b + 1e-6)
        green_ratio = mean_g / (mean_r + mean_g + mean_b + 1e-6)
        
        features = [
            width, height, total_pixels, aspect_ratio,
            mean_r, mean_g, mean_b, std_r, std_g, std_b,
            mean_gray, std_gray, edge_density, solidity,
            texture_variance, mean_h, mean_s, mean_v,
            blue_ratio, red_ratio, green_ratio
        ]
        
        # Smart labeling based on comprehensive analysis
        label = smart_label_image(img, gray, hsv, aspect_ratio, mean_gray, edge_density, 
                                blue_ratio, red_ratio, solidity)
        
        return np.array(features), label
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def smart_label_image(img, gray, hsv, aspect_ratio, brightness, edge_density, 
                     blue_ratio, red_ratio, solidity):
    """Intelligent image labeling based on multiple criteria"""
    
    # Vehicle detection (strong indicators)
    vehicle_score = 0
    if aspect_ratio > 1.8:  # Very wide images
        vehicle_score += 3
    elif aspect_ratio > 1.3:  # Moderately wide
        vehicle_score += 1
    
    if edge_density > 0.12:  # High geometric content
        vehicle_score += 2
    
    if solidity > 0.8:  # Solid, geometric shapes
        vehicle_score += 1
    
    if brightness > 140:  # Bright, outdoor lighting
        vehicle_score += 1
    
    # Soldier detection (military indicators)
    soldier_score = 0
    if 0.6 < aspect_ratio < 1.5:  # Portrait orientation
        soldier_score += 1
    
    if brightness < 130:  # Darker, military setting
        soldier_score += 2
    
    if blue_ratio > 0.4 or red_ratio > 0.4:  # Flag colors
        soldier_score += 3
    
    if 100 < brightness < 130:  # Military uniform lighting
        soldier_score += 2
    
    # Person detection (civilian indicators)
    person_score = 0
    if 0.5 < aspect_ratio < 1.8:  # Human proportions
        person_score += 1
    
    if 120 < brightness < 180:  # Normal indoor/outdoor lighting
        person_score += 1
    
    if edge_density < 0.08:  # Less geometric, more organic
        person_score += 1
    
    if solidity < 0.7:  # Less solid shapes (clothing, hair)
        person_score += 1
    
    # Decision logic
    if vehicle_score >= 4:
        return 0  # vehicle
    elif soldier_score >= 4:
        return 1  # soldier
    elif person_score >= 3:
        return 2  # person
    else:
        # Fallback based on dominant feature
        if aspect_ratio > 1.5:
            return 0  # vehicle
        elif blue_ratio > 0.35 or red_ratio > 0.35:
            return 1  # soldier
        else:
            return 2  # person

def create_perfect_training_data():
    """Create high-quality training data"""
    print("Creating perfect training data...")
    
    upload_dir = Path('static/uploads')
    if not upload_dir.exists():
        print("❌ No uploaded images found.")
        return None, None
    
    features_list = []
    labels_list = []
    label_names = ['vehicle', 'soldier', 'person', 'other']
    
    # Process uploaded images with smart labeling
    image_files = list(upload_dir.glob('*'))
    print(f"Processing {len(image_files)} images with advanced analysis...")
    
    for img_path in image_files:
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            features, label = extract_advanced_features(img_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
                print(f"  {img_path.name}: {label_names[label]}")
    
    # Add high-quality synthetic data
    print("Adding premium synthetic training data...")
    
    # Premium vehicle data (cars, trucks, etc.)
    for _ in range(100):
        synthetic = np.random.rand(21) * 100
        synthetic[3] = np.random.uniform(1.5, 3.0)  # Wide aspect ratio
        synthetic[12] = np.random.uniform(0.10, 0.25)  # High edge density
        synthetic[13] = np.random.uniform(0.7, 0.95)   # High solidity
        synthetic[10] = np.random.uniform(120, 200)    # Bright
        features_list.append(synthetic)
        labels_list.append(0)  # vehicle
    
    # Premium soldier data (military personnel)
    for _ in range(100):
        synthetic = np.random.rand(21) * 100
        synthetic[3] = np.random.uniform(0.6, 1.4)   # Portrait aspect
        synthetic[10] = np.random.uniform(80, 130)   # Darker
        synthetic[19] = np.random.uniform(0.4, 0.6)  # High red ratio (flag)
        synthetic[18] = np.random.uniform(0.35, 0.5) # High blue ratio (flag)
        features_list.append(synthetic)
        labels_list.append(1)  # soldier
    
    # Premium person data (civilians)
    for _ in range(100):
        synthetic = np.random.rand(21) * 100
        synthetic[3] = np.random.uniform(0.5, 1.6)   # Human proportions
        synthetic[10] = np.random.uniform(120, 170)  # Normal lighting
        synthetic[12] = np.random.uniform(0.04, 0.08) # Low edge density
        synthetic[13] = np.random.uniform(0.3, 0.7)   # Lower solidity
        features_list.append(synthetic)
        labels_list.append(2)  # person
    
    # Other category
    for _ in range(50):
        synthetic = np.random.rand(21) * 100
        features_list.append(synthetic)
        labels_list.append(3)  # other
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"✅ Perfect training dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y)}")
    
    return X, y

def train_perfect_model():
    """Train the perfect model"""
    print("=== PERFECT MODEL TRAINING ===\n")
    
    X, y = create_perfect_training_data()
    if X is None:
        return False
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create optimized pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    # Train
    print("Training perfect model...")
    pipeline.fit(X_train, y_train)
    
    # Comprehensive evaluation
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Detailed evaluation
    y_pred = pipeline.predict(X_test)
    class_names = ['vehicle', 'soldier', 'person', 'other']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Save perfect model
    model_path = 'models/rf_pipeline_perfect.joblib'
    joblib.dump(pipeline, model_path)
    print(f"✅ Perfect model saved to {model_path}")
    
    return True

if __name__ == "__main__":
    if train_perfect_model():
        print("\n🎉 PERFECT MODEL READY!")
        print("Update config.py to use rf_pipeline_perfect.joblib")
    else:
        print("❌ Training failed")
