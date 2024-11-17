from flask import Flask, render_template, request, jsonify
import base64
import os

app = Flask(__name__)

# Folder to save images
SAVE_FOLDER = 'captured_images'
os.makedirs(SAVE_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apply')
def apply():
    return render_template('apply.html')

def save_image_with_incremented_name(name, image_data):
    # Normalize the name to create a valid filename
    base_filename = f"{name.replace(' ', '_')}_image"
    filename = base_filename + ".png"
    filepath = os.path.join(SAVE_FOLDER, filename)
    counter = 1

    # Increment filename if it already exists
    while os.path.exists(filepath):
        filename = f"{base_filename}_{counter}.png"
        filepath = os.path.join(SAVE_FOLDER, filename)
        counter += 1

    # Save the image
    with open(filepath, 'wb') as f:
        f.write(image_data)

    return filename

@app.route('/save-image', methods=['POST'])
def save_image():
    data = request.json
    image_data = data['image']
    name = data['name']

    # Decode the image data
    image_data = base64.b64decode(image_data.split(',')[1])

    # Save the image with a unique name
    saved_filename = save_image_with_incremented_name(name, image_data)

    return jsonify({"status": "success", "message": f"Image saved as {saved_filename}"}), 200

if __name__ == '__main__':
    app.run(debug=True)