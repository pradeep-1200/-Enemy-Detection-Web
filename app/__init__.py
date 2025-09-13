from flask import Flask
from config import Config
import os

def create_app():
    # Get absolute path to project root
    basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Create Flask app with explicit template and static folders
    app = Flask(__name__, 
                template_folder=os.path.join(basedir, 'templates'),
                static_folder=os.path.join(basedir, 'static'))
    
    app.config.from_object(Config)
    
    # Ensure upload folder exists
    upload_path = os.path.join(basedir, 'static', 'uploads')
    os.makedirs(upload_path, exist_ok=True)
    
    # Print paths for debugging
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    print(f"Templates exist: {os.path.exists(app.template_folder)}")
    
    from app.routes import main
    app.register_blueprint(main)
    
    return app
