from flask import Flask, render_template, request, jsonify, url_for, flash, redirect, Blueprint
from app.models import detection_model
from app.utils import save_uploaded_file, cleanup_uploaded_file
import os
from pathlib import Path

# Get absolute paths
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'

# Create Flask app with absolute paths
app = Flask(__name__, 
            template_folder=str(TEMPLATE_DIR),
            static_folder=str(STATIC_DIR))

# Configuration
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload directory
os.makedirs('static/uploads', exist_ok=True)

print(f"Template folder: {app.template_folder}")
print(f"Static folder: {app.static_folder}")
print(f"Templates directory exists: {TEMPLATE_DIR.exists()}")

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload')
def upload():
    """Upload page"""
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(url_for('upload'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('upload'))
        
        # Save the file
        filename = save_uploaded_file(file)
        if not filename:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
            return redirect(url_for('upload'))
        
        # Get full path
        filepath = os.path.join('static/uploads', filename)
        
        # Make prediction
        result = detection_model.predict(filepath)
        
        if 'error' in result:
            flash(f'Prediction error: {result["error"]}')
            cleanup_uploaded_file(filename)
            return redirect(url_for('upload'))
        
        # Render results
        return render_template('results.html', 
                             result=result, 
                             image_url=url_for('static', filename=f'uploads/{filename}'),
                             filename=filename)
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('upload'))

@app.route('/cleanup/<filename>')
def cleanup(filename):
    """Cleanup uploaded file"""
    cleanup_uploaded_file(filename)
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Verify templates exist before starting
    if not TEMPLATE_DIR.exists():
        print("ERROR: Templates directory not found!")
        exit(1)
    
    print(f"Starting server with templates at: {TEMPLATE_DIR}")
    app.run(debug=True, host='0.0.0.0', port=5000)
