import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store features for comparison
image_features = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_hog_features(image_path):
    """Extract HOG features from an image"""
    try:
        # Load image
        image = imread(image_path)
        
        # Convert to grayscale if it's a color image
        if image.ndim == 3:
            # Use the mean of all channels to convert to grayscale
            image = np.mean(image, axis=2)
        
        # Resize image to a consistent size
        image = resize(image, (128, 128), anti_aliasing=True)
        
        # Extract HOG features
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        
        # Normalize the feature vector
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

def load_existing_features():
    """Load features for existing images in the uploads folder"""
    if not os.path.exists(UPLOAD_FOLDER):
        return
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                features = extract_hog_features(filepath)
                image_features[filename] = features
                print(f"Loaded features for {filename}")
            except Exception as e:
                print(f"Error loading features for {filename}: {e}")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists('templates'):
    os.makedirs('templates')

# Load existing features when the app starts
load_existing_features()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) if file.filename else 'unnamed'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features
        try:
            features = extract_hog_features(filepath)
            image_features[filename] = features
            return jsonify({'filename': filename, 'uploaded': True})
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/compare', methods=['POST'])
def compare_images():
    data = request.get_json()
    image1 = data.get('image1')
    image2 = data.get('image2')
    
    if image1 not in image_features or image2 not in image_features:
        return jsonify({'error': 'One or both images not found'})
    
    try:
        # Get feature vectors
        features1 = image_features[image1].reshape(1, -1)
        features2 = image_features[image2].reshape(1, -1)
        
        # Calculate cosine similarity
        similarity_score = cosine_similarity(features1, features2)[0][0]
        
        # Ensure the similarity is in the range [0, 1]
        similarity_score = max(0, min(1, similarity_score))
        
        # Determine verdict based on similarity score
        if similarity_score > 0.75:
            verdict = "Same Person / Highly Similar"
        elif 0.55 < similarity_score <= 0.75:
            verdict = "Possibly Similar"
        else:
            verdict = "Different Faces"
        
        return jsonify({
            'image1': image1,
            'image2': image2,
            'similarity_score': float(similarity_score),
            'percentage': float(similarity_score * 100),
            'verdict': verdict
        })
    except Exception as e:
        return jsonify({'error': f'Error comparing images: {str(e)}'})

@app.route('/uploads')
def list_uploads():
    if not os.path.exists(UPLOAD_FOLDER):
        return jsonify({'images': []})
    
    images = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            images.append(filename)
    
    return jsonify({'images': images})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        file_deleted = False
        feature_deleted = False
        
        # Remove from image_features dictionary
        if filename in image_features:
            del image_features[filename]
            feature_deleted = True
        
        # Remove file from uploads directory
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            file_deleted = True
        
        if file_deleted or feature_deleted:
            return jsonify({'message': f'File {filename} deleted successfully'})
        else:
            return jsonify({'error': f'File {filename} not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error deleting file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)