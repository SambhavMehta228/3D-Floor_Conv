import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


class_names = ['BG', 'building', 'door', 'window', 'wall']

# Load Mask R-CNN model
def load_model():
    return keras.models.load_model('mask_rcnn_coco.h5')  

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Denoise image
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    
    # Equalize histogram to improve contrast
    equalized = cv2.equalizeHist(denoised)
    
    return equalized

def detect_objects(model, image):
    input_image = cv2.resize(image, (1024, 1024))
    input_image = input_image.astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)
    
    detections = model.detect(input_image, verbose=1)
    
    return detections[0]

def create_floor_plan(image, detections):
    floor_plan = np.zeros_like(image)
    
    for i, class_id in enumerate(detections['class_ids']):
        mask = detections['masks'][:, :, i]
        class_name = class_names[class_id]
        
        if class_name in ['building', 'door', 'window', 'wall']:
            color = (255, 255, 255)  
            floor_plan[mask] = color
    
    return floor_plan

def convert_3d_to_floor_plan(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Load the Mask R-CNN model
    model = load_model()
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Detect objects
    detections = detect_objects(model, preprocessed_image)
    
    # Create the floor plan
    floor_plan = create_floor_plan(preprocessed_image, detections)
    
    # Save the floor plan
    cv2.imwrite(output_path, floor_plan)
    print(f"Floor plan saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_3d_to_floor_plan.py <input_image_path> <output_image_path>")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        convert_3d_to_floor_plan(input_image_path, output_image_path)
 