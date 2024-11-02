import numpy as np
import cv2
from tensorflow.keras.models import load_model
 
model = load_model('mnist_cnn_model.h5')
print("Model loaded successfully!")

 
def segment_and_preprocess_image(image_path):
   
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
 
    _, img_thresh = cv2.threshold(img_blur, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

 
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []
     
    bounding_boxes = sorted([cv2.boundingRect(contour) for contour in contours], key=lambda x: x[0])

    for (x, y, w, h) in bounding_boxes:
       
        digit_img = img_thresh[y:y + h, x:x + w]
        
        
        aspect_ratio = w / h
        if aspect_ratio > 1:  
            new_w = 20
            new_h = int(20 / aspect_ratio)
        else:   
            new_h = 20
            new_w = int(20 * aspect_ratio)

        digit_img = cv2.resize(digit_img, (new_w, new_h))
        
       
        top_pad = (28 - new_h) // 2
        bottom_pad = 28 - new_h - top_pad
        left_pad = (28 - new_w) // 2
        right_pad = 28 - new_w - left_pad

        padded_img = cv2.copyMakeBorder(digit_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
        padded_img = padded_img.astype("float32") / 255.0
        padded_img = np.reshape(padded_img, (28, 28, 1))

        digit_images.append(padded_img)

    return np.array(digit_images)
 
image_path = 'hh.jpg'   

 
digit_images = segment_and_preprocess_image(image_path)
 
if digit_images.size == 0:
    print("No digits found in the image.")
else:
    predictions = model.predict(digit_images)
 
    predicted_labels = [(np.argmax(pred), np.max(pred)) for pred in predictions]
    print("Predicted Labels and Confidences:", predicted_labels)
