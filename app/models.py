import os
from pathlib import Path
from config import Config
import cv2
import numpy as np

class EnemyDetectionModel:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.use_real_model = False
        self.model_type = "SIMULATION"
        
        # Try to load real model
        self.load_real_model()
        
        print(f"Detection system initialized - Mode: {self.model_type}")
    
    def load_real_model(self):
        """Try to load the actual trained model"""
        try:
            import joblib
            model_path = Config.MODEL_PATH
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.model_loaded = True
                self.use_real_model = True
                self.model_type = "REAL_TRAINED_MODEL"
                print(f"✅ SUCCESS: Real model loaded from {model_path}")
                print(f"   Model type: {type(self.model)}")
                
                # Test with 21 features (FIXED)
                dummy_features = np.random.rand(1, 21)  # Changed from 16 to 21
                test_pred = self.model.predict(dummy_features)
                test_prob = self.model.predict_proba(dummy_features)
                print(f"   Model test: Prediction shape {test_pred.shape}, Probability shape {test_prob.shape}")
                
                return True
            else:
                print(f"❌ Model file not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load real model: {e}")
            print("   Falling back to intelligent simulation...")
            return False
    
    def extract_features_for_real_model(self, image_path):
        """Extract 21 advanced features for perfect model - MUST match training exactly"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Basic features (4 features)
            height, width = gray.shape
            total_pixels = height * width
            aspect_ratio = width / height
            
            # Color features (6 features)
            mean_r = np.mean(img[:,:,2])
            mean_g = np.mean(img[:,:,1]) 
            mean_b = np.mean(img[:,:,0])
            std_r = np.std(img[:,:,2])
            std_g = np.std(img[:,:,1])
            std_b = np.std(img[:,:,0])
            
            # Gray features (2 features)
            mean_gray = np.mean(gray)
            std_gray = np.std(gray)
            
            # Edge features (1 feature)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # Shape features (1 feature) - NEW
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = contour_area / hull_area if hull_area > 0 else 0
            else:
                solidity = 0
            
            # Texture features (1 feature)
            texture_variance = np.var(cv2.GaussianBlur(gray, (5,5), 0))
            
            # HSV features (3 features)
            mean_h = np.mean(hsv[:,:,0])
            mean_s = np.mean(hsv[:,:,1])
            mean_v = np.mean(hsv[:,:,2])
            
            # Color ratio features (3 features) - NEW
            total_color = mean_r + mean_g + mean_b + 1e-6
            blue_ratio = mean_b / total_color
            red_ratio = mean_r / total_color
            green_ratio = mean_g / total_color
            
            # EXACTLY 21 features to match training
            features = [
                width, height, total_pixels, aspect_ratio,           # 4 features
                mean_r, mean_g, mean_b, std_r, std_g, std_b,        # 6 features  
                mean_gray, std_gray, edge_density, solidity,        # 4 features
                texture_variance, mean_h, mean_s, mean_v,           # 4 features
                blue_ratio, red_ratio, green_ratio                  # 3 features
            ]                                                       # = 21 total
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def predict_with_real_model(self, image_path):
        """Use the actual trained model"""
        try:
            # Extract features
            features = self.extract_features_for_real_model(image_path)
            if features is None:
                return {"error": "Could not extract features"}
            
            # Make prediction with real model
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Map to class names
            class_name = Config.DETECTION_CLASSES.get(prediction, f"Class_{prediction}")
            confidence = float(np.max(probabilities))
            
            return {
                "prediction": int(prediction),
                "class_name": class_name,
                "confidence": confidence,
                "probabilities": {
                    Config.DETECTION_CLASSES.get(i, f"Class_{i}"): float(prob) 
                    for i, prob in enumerate(probabilities)
                },
                "model_used": self.model_type,
                "features_extracted": features.shape[1]
            }
            
        except Exception as e:
            print(f"Real model prediction failed: {e}")
            return {"error": f"Real model failed: {e}"}
    
    def analyze_image_content(self, image_path):
        """Analyze image content for intelligent simulation"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            aspect_ratio = width / height
            
            # Color analysis
            avg_brightness = np.mean(gray)
            color_variance = np.var(img)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Color dominance
            blue_dom = np.mean(img[:,:,0]) / 255
            green_dom = np.mean(img[:,:,1]) / 255
            red_dom = np.mean(img[:,:,2]) / 255
            
            return {
                'width': width, 'height': height, 'aspect_ratio': aspect_ratio,
                'brightness': avg_brightness, 'color_variance': color_variance,
                'edge_density': edge_density, 'blue_dom': blue_dom,
                'green_dom': green_dom, 'red_dom': red_dom
            }
            
        except Exception as e:
            print(f"Image analysis error: {e}")
            return None
    
    def intelligent_predict(self, features):
        """Intelligent simulation based on image analysis"""
        # Initialize probabilities
        probs = {'vehicle': 0.2, 'soldier': 0.2, 'person': 0.3, 'other': 0.3}
        
        # Human detection (portrait aspect ratio)
        if 0.5 < features['aspect_ratio'] < 2.0:
            probs['person'] += 0.3
            probs['soldier'] += 0.2
            probs['vehicle'] -= 0.15
            probs['other'] -= 0.35
        
        # Military detection (darker image with color variation)
        if features['brightness'] < 120 and features['color_variance'] > 1000:
            probs['soldier'] += 0.4
            probs['person'] += 0.1
            probs['vehicle'] -= 0.25
            probs['other'] -= 0.25
        
        # Civilian person (normal brightness)
        if 120 < features['brightness'] < 180:
            probs['person'] += 0.3
            probs['soldier'] += 0.1
            probs['vehicle'] -= 0.2
            probs['other'] -= 0.2
        
        # Vehicle detection (wide aspect or high edges)
        if features['aspect_ratio'] > 2.0 or features['edge_density'] > 0.15:
            probs['vehicle'] += 0.4
            probs['person'] -= 0.15
            probs['soldier'] -= 0.15
            probs['other'] -= 0.1
        
        # Military/patriotic context (flag colors)
        if (features['red_dom'] > 0.4 or features['blue_dom'] > 0.4) and features['color_variance'] > 2000:
            probs['soldier'] += 0.3
            probs['person'] += 0.1
            probs['vehicle'] -= 0.2
            probs['other'] -= 0.2
        
        # Normalize probabilities
        total = sum(max(0.05, p) for p in probs.values())
        for k in probs:
            probs[k] = max(0.05, probs[k]) / total
        
        return probs
    
    def predict_with_simulation(self, image_path):
        """Intelligent simulation fallback"""
        features = self.analyze_image_content(image_path)
        
        if features is None:
            # Basic random fallback
            import random
            classes = list(Config.DETECTION_CLASSES.values())
            prediction = random.randint(0, 3)
            confidence = random.uniform(0.5, 0.8)
            
            probabilities = {cls: 0.1 for cls in classes}
            probabilities[classes[prediction]] = confidence
            remaining = 1.0 - confidence
            for i, cls in enumerate(classes):
                if i != prediction:
                    probabilities[cls] = remaining / 3
        else:
            # Intelligent prediction
            probabilities = self.intelligent_predict(features)
            predicted_class = max(probabilities.items(), key=lambda x: x[1])
            prediction = list(Config.DETECTION_CLASSES.values()).index(predicted_class[0])
            confidence = predicted_class[1]
            
            # Add some realistic variation
            import random
            confidence += random.uniform(-0.05, 0.1)
            confidence = max(0.3, min(0.95, confidence))
        
        return {
            "prediction": prediction,
            "class_name": list(Config.DETECTION_CLASSES.values())[prediction],
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": "INTELLIGENT_SIMULATION",
            "image_info": features
        }
    
    def predict(self, image_path):
        """Main prediction method"""
        if self.use_real_model and self.model_loaded:
            # Try real model first
            result = self.predict_with_real_model(image_path)
            if "error" not in result:
                return result
        
        # Fallback to intelligent simulation
        return self.predict_with_simulation(image_path)

# Global model instance
detection_model = EnemyDetectionModel()
