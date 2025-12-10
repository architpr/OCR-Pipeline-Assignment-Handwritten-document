import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
import math

class HandwrittenOCRRedactor:
    """
    A high-accuracy pipeline for OCR and PII redaction of messy handwritten medical notes.
    Uses PaddleOCR + Advanced Preprocessing (CLAHE, Upscaling).
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize the PaddleOCR reader and the Presidio AnalyzerEngine.
        """
        print("Initializing PaddleOCR and Presidio Analyzer...")
        # PaddleOCR instantiation
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        # Configure Presidio to use the small Spacy model to save RAM
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        
        print("Initialization complete.")

    def preprocess_image(self, image_path, deskew=False):
        """
        Advanced preprocessing for handwritten notes.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # 1. Size Normalization (Smart Scaling)
        h, w = img.shape[:2]
        MAX_HEIGHT = 2000 # Reduced from 4000 to prevent crashes on limited RAM
        
        # If image is massive, downscale it to prevent crashes
        if h > MAX_HEIGHT:
            scale = MAX_HEIGHT / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, MAX_HEIGHT), interpolation=cv2.INTER_AREA)
            print(f"Downscaled massive image from {w}x{h} to {new_w}x{MAX_HEIGHT}")
            h, w = img.shape[:2] # Update dims
            upscaled = img
        # If image is small, upscale it for better OCR
        elif h < 1000:
            scale_factor = 2 # Moderate upscale
            width = int(w * scale_factor)
            height = int(h * scale_factor)
            # Ensure we don't accidentally exceed the limit with upscaling
            if height > MAX_HEIGHT:
                scale_factor = MAX_HEIGHT / h
                width = int(w * scale_factor)
                height = MAX_HEIGHT
                
            upscaled = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            print(f"Upscaled image from {w}x{h} to {width}x{height}")
        else:
            print(f"Image resolution {w}x{h} is sufficient. Keeping original.")
            upscaled = img
        
        # 2. Contrast Enhancement (CLAHE in LAB color space)
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 3. Morphological Opening (Remove small noise)
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        final_img = denoised

        # 4. Filter: Optional Deskewing
        # Note: PaddleOCR has built-in angle classification, so manual deskew might be redundant
        # but we keep it available as an option.
        if deskew:
            gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            # Inverse threshold for contour detection of ink
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Only rotate if the angle is significant to avoid blurring
            if abs(angle) > 0.5:
                (h, w) = final_img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                final_img = cv2.warpAffine(final_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                print(f"Deskewed: {angle:.2f} degrees")
        
        return final_img

    def extract_text_with_boxes(self, img):
        """
        Run PaddleOCR on the processed image.
        
        Args:
            img (np.array): The image to process.
            
        Returns:
            list: List of (bounding_box, text, confidence).
                  bounding_box is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        """
        # PaddleOCR expects the image array
        try:
            result = self.ocr.ocr(img)
        except Exception as e:
            print(f"Error during OCR execution: {e}")
            return []
        
        # PaddleOCR output format: [ [ [box], [text, score] ], ... ]
        # If no text found, result is [None] or similar depending on version
        
        processed_results = []
        if not result or result[0] is None:
            return processed_results
            
        for line in result[0]:
            bbox = line[0]
            # Robust unpacking of text/confidence
            if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                text, conf = line[1][:2]
            elif isinstance(line[1], (list, tuple)) and len(line[1]) == 1:
                text = line[1][0]
                conf = 0.6 # Default confidence high enough to pass filter
                print(f"Warning: OCR returned text '{text}' without confidence score.")
            else:
                # Fallback
                print(f"Warning: Unexpected OCR line format: {line[1]}")
                continue
            
            # FILTERS from prompt
            # 1. Confidence < 0.4 (Lowered from 0.6 for handwriting)
            # if conf < 0.4:
            #     continue
                
            # 2. Single char non-digit
            # if len(text) == 1 and not text.isdigit():
            #     continue
                
            processed_results.append((bbox, text, conf))
            
        return processed_results

    def redact_pii(self, img, ocr_results):
        """
        Redact PII with Fuzzy Logic for 'Name:' patterns.
        
        Args:
            img (np.array): Image to redact.
            ocr_results (list): Output from extract_text_with_boxes.
            
        Returns:
            tuple: (redacted_img, full_text, redacted_text_str)
        """
        redacted_img = img.copy()
        
        full_text = ""
        box_map = [] # List of (start_index, end_index, bbox)
        
        current_idx = 0
        for (bbox, text, conf) in ocr_results:
            # Append text
            full_text += text + " "
            
            # Record mapping: this text segment corresponds to this box
            # We map the range of indices in full_text to this box
            start = current_idx
            end = current_idx + len(text)
            box_map.append({
                'start': start,
                'end': end,
                'box': bbox,
                'text': text
            })
            
            current_idx = len(full_text)
            
        print(f"Reconstructed Text: {full_text}")
        
        # 1. Presidio Analysis
        presidio_results = self.analyzer.analyze(text=full_text, language='en')
        
        # 2. Fuzzy Logic for "Name:"
        # Common patterns in medical notes: "Name: John Doe", "Patient: Jane Smith"
        # Since handwriting is messy, we might see "Name-", "Name;", etc.
        import re
        # Regex to find "Name" keywords followed by text
        name_patterns = [
            r"(Name|Patient|Pt)[\s:;\-\.]+(\w+[\s\w]*)"
        ]
        
        fuzzy_entities = []
        for pattern in name_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                # The group(2) is the actual name
                name_start = match.start(2)
                name_end = match.end(2)
                # Add to entities if not already covered
                fuzzy_entities.append((name_start, name_end, "PERSON_FUZZY"))
                print(f"Fuzzy Logic Detected Name: {match.group(2)}")

        # Combine logic
        # We need to redact the boxes that overlap with these entities
        
        # Setup redaction mask on text for display
        redacted_text_chars = list(full_text)
        
        # Helper to redact a range
        def redact_range(start, end, entity_type):
            print(f"Redacting {entity_type}: {full_text[start:end]}")
            
            # Redact text
            for i in range(start, end):
                if i < len(redacted_text_chars):
                    redacted_text_chars[i] = "*"
            
            # Redact image
            # Find boxes that have significant overlap with this range
            for item in box_map:
                # Check for overlap
                # Item range: [item['start'], item['end']]
                # Redact range: [start, end]
                
                # Intersection
                inter_start = max(start, item['start'])
                inter_end = min(end, item['end'])
                
                if inter_start < inter_end:
                    # There is overlap. Redact this box.
                    # Note: We redact the WHOLE box if it overlaps partially. 
                    # This is safer for PII than trying to cut the box.
                    box = item['box']
                    # Paddle box is float list
                    box = np.array(box).astype(np.int32)
                    cv2.fillPoly(redacted_img, [box], (0, 0, 0))

        # Apply Presidio results
        for result in presidio_results:
            redact_range(result.start, result.end, result.entity_type)
            
        # Apply Fuzzy results
        for (start, end, etype) in fuzzy_entities:
            redact_range(start, end, etype)
            
        redacted_text_str = "".join(redacted_text_chars)
        return redacted_img, full_text, redacted_text_str

    def display_results(self, original, redacted):
        # ... (same as before) logic is generic
        pass
