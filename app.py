from PIL import Image
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
 
image_path_1 = "s1.jpg"
 
image = cv2.imread(image_path_1)
 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

_, thresh = cv2.threshold(opened, 120, 255, cv2.THRESH_BINARY_INV)

custom_config = r'--oem 3 --psm 10 outputbase digits'

digits = pytesseract.image_to_string(thresh, config=custom_config)

print("Detected digits:", digits)