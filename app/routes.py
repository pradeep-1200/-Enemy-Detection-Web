from flask import Blueprint, render_template, request, jsonify, url_for, flash, redirect
from app.models import detection_model
from app.utils import save_uploaded_file, cleanup_uploaded_file
import os

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@main.route('/upload')
def upload():
    """Upload page"""
    return render_template('upload.html')

@main.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(url_for('main.upload'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(url_for('main.upload'))
        
        # Save the file
        filename = save_uploaded_file(file)
        if not filename:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, or GIF files.')
            return redirect(url_for('main.upload'))
        
        # Get full path
        filepath = os.path.join('static/uploads', filename)
        
        # Make prediction
        result = detection_model.predict(filepath)
        
        if 'error' in result:
            flash(f'Prediction error: {result["error"]}')
            cleanup_uploaded_file(filename)
            return redirect(url_for('main.upload'))
        
        # Render results
        return render_template('results.html', 
                             result=result, 
                             image_url=url_for('static', filename=f'uploads/{filename}'),
                             filename=filename)
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('main.upload'))

@main.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = save_uploaded_file(file)
        if not filename:
            return jsonify({'error': 'Invalid file type'}), 400
        
        filepath = os.path.join('static/uploads', filename)
        result = detection_model.predict(filepath)
        
        # Cleanup
        cleanup_uploaded_file(filename)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/cleanup/<filename>')
def cleanup(filename):
    """Cleanup uploaded file"""
    cleanup_uploaded_file(filename)
    return redirect(url_for('main.index'))
