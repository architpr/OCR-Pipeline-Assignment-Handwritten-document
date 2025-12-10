import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

class HandwrittenOCR:
    """
    A robust, object-oriented pipeline for OCR of Handwritten Medical Notes.
    Uses EasyOCR for text detection and Microsoft TrOCR for recognition.
    """

    def __init__(self):
        """
        Initialize:
        1. EasyOCR Reader (Detection only).
        2. Microsoft TrOCR (Recognition).
        """
        print("Initializing EasyOCR (for detection)...")
        # Initialize EasyOCR just for detection
        self.reader = easyocr.Reader(['en'], gpu=True)
        
        print("Initializing Microsoft TrOCR (for recognition)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load TrOCR model and processor
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
        
        print("Initialization complete.")

    def preprocess_image(self, image_path):
        """
        Load and preprocess the image:
        1. Grayscale
        2. Gaussian Blur
        3. Otsu's Thresholding
        4. Deskewing
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            np.array: The preprocessed image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load image
        img = cv2.imread(image_path)
        
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Otsu's Thresholding to binarize
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Deskewing Logic (on inverted binary)
        _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        coords = np.column_stack(np.where(thresh_inv > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle format
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate if angle is significant
        if abs(angle) > 0.5:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            print(f"Deskewed image by {angle:.2f} degrees.")
            return img_rotated
            
        return img

    def recognize_trocr(self, img_crop):
        """
        Use TrOCR to recognize text from a cropped image segment.
        """
        # Convert CV2 (BGR) to PIL (RGB)
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb).convert("RGB")
        
        # Prepare input
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        generated_ids = self.model.generate(pixel_values, max_new_tokens=20) # Max tokens limit to prevent hanging on noise
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

    def sort_by_line(self, ocr_results, y_threshold=20):
        """
        Sort OCR results into lines based on Y-coordinates.
        Results are sorted top-to-bottom, then left-to-right.
        """
        if not ocr_results:
            return []
            
        # Initial sort by Top-Left Y
        ocr_results.sort(key=lambda x: x[0][0][1])
        
        lines = []
        current_line = [ocr_results[0]]
        
        for i in range(1, len(ocr_results)):
            bbox = ocr_results[i][0]
            prev_bbox = current_line[-1][0]
            
            y_curr = bbox[0][1]
            y_prev = prev_bbox[0][1]
            
            # If close in Y, same line
            if abs(y_curr - y_prev) < y_threshold:
                current_line.append(ocr_results[i])
            else:
                # New line, sort previous line by X and add
                current_line.sort(key=lambda x: x[0][0][0])
                lines.extend(current_line)
                current_line = [ocr_results[i]]
        
        # Add last line
        current_line.sort(key=lambda x: x[0][0][0])
        lines.extend(current_line)
        
        return lines

    def extract_text_with_boxes(self, img):
        """
        Run Detection (EasyOCR) + Recognition (TrOCR).
        """
        print("Detecting text boxes...")
        # Use EasyOCR to get boxes. 
        # detection_results has (bbox, text, conf)
        # We will IGNORE the text from EasyOCR and re-run TrOCR on the crop
        detection_results = self.reader.readtext(img)
        
        final_results = []
        print(f"Found {len(detection_results)} text segments. Recognizing with TrOCR...")
        
        for i, (bbox, _, conf) in enumerate(detection_results):
            # Crop the image at the bounding box
            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # We need min_x, min_y, max_x, max_y
            np_box = np.array(bbox)
            min_x = int(np.min(np_box[:, 0]))
            max_x = int(np.max(np_box[:, 0]))
            min_y = int(np.min(np_box[:, 1]))
            max_y = int(np.max(np_box[:, 1]))
            
            # Clip to image bounds
            h, w = img.shape[:2]
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(w, max_x)
            max_y = min(h, max_y)
            
            # Logic check
            if max_x - min_x < 5 or max_y - min_y < 5:
                continue # Too small
                
            crop = img[min_y:max_y, min_x:max_x]
            
            # Run TrOCR
            try:
                text = self.recognize_trocr(crop)
                # print(f"Segment {i}: {text}") # Debug
                final_results.append((bbox, text, conf))
            except Exception as e:
                print(f"TrOCR Error on segment {i}: {e}")
                
        # Post-process: Sort by line
        sorted_results = self.sort_by_line(final_results)
            
        return sorted_results

    def get_annotated_image(self, original_img, ocr_results):
        """
        Draw bounding boxes and text on the image and return it.
        """
        annotated_img = original_img.copy()
        
        for (bbox, text, conf) in ocr_results:
            pts = np.array(bbox, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Put text (using first point of bbox)
            cv2.putText(annotated_img, text, (pts[0][0][0], pts[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated_img

    def display_results(self, original_img, ocr_results):
        annotated_img = self.get_annotated_image(original_img, ocr_results)
        
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.title("Original Pre-processed")
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Annotated TrOCR Output")
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ocr = HandwrittenOCR()
    
    # Use a sample image path
    # Try finding an existing one or default to 'sample_doctor_note.jpg'
    sample_image = "sample_doctor_note.jpg"
    
    # Check if sample exists, if not use one of the uploaded images if available
    if not os.path.exists(sample_image):
        # Look for uploaded images in the directory
        files = [f for f in os.listdir('.') if f.startswith('uploaded_')]
        if files:
            sample_image = files[0]
            print(f"Sample image not found, using {sample_image}")
        else:
            print(f"Warning: {sample_image} not found. Please provide a valid image.")
    
    if os.path.exists(sample_image):
        print(f"Processing {sample_image}...")
        try:
            processed_img = ocr.preprocess_image(sample_image)
            results = ocr.extract_text_with_boxes(processed_img)
            
            print("\nExtracted Text:")
            for _, text, conf in results:
                print(f"[{conf:.2f}] {text}")
                
            ocr.display_results(processed_img, results)
        except Exception as e:
            print(f"Error: {e}")
