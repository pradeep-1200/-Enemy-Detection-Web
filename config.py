import os
from pathlib import Path

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    BASE_DIR = Path(__file__).parent
    
    # Priority order for models
    PERFECT_MODEL_PATH = BASE_DIR / 'models' / 'rf_pipeline_perfect.joblib'
    SMART_MODEL_PATH = BASE_DIR / 'models' / 'rf_pipeline_smart.joblib'
    
    if PERFECT_MODEL_PATH.exists():
        MODEL_PATH = PERFECT_MODEL_PATH
        print(f"🎯 Using PERFECT model: {MODEL_PATH}")
    elif SMART_MODEL_PATH.exists():
        MODEL_PATH = SMART_MODEL_PATH
        print(f"🧠 Using SMART model: {MODEL_PATH}")
    else:
        MODEL_PATH = BASE_DIR / 'models' / 'rf_pipeline_compatible.joblib'
        print(f"⚠️ Using compatible model: {MODEL_PATH}")
    
    DETECTION_CLASSES = {
        0: 'vehicle',
        1: 'soldier',
        2: 'person',
        3: 'other'
    }
