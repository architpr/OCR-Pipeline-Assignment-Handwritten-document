import os
import cv2
from handwritten_ocr_redactor import HandwrittenOCRRedactor

def process_samples():
    # Absolute paths to the uploaded images
    image_paths = [
        r"C:/Users/ACER/.gemini/antigravity/brain/b0f494fe-a143-43ce-8082-d163871f753e/uploaded_image_0_1765352314219.jpg",
        r"C:/Users/ACER/.gemini/antigravity/brain/b0f494fe-a143-43ce-8082-d163871f753e/uploaded_image_1_1765352314219.jpg",
        r"C:/Users/ACER/.gemini/antigravity/brain/b0f494fe-a143-43ce-8082-d163871f753e/uploaded_image_2_1765352314219.jpg"
    ]

    pipeline = HandwrittenOCRRedactor()

    for i, img_path in enumerate(image_paths):
        print(f"Processing {img_path}...")
        try:
            if not os.path.exists(img_path):
                print(f"Error: File not found: {img_path}")
                continue

            # Step 1: Preprocess
            processed_img = pipeline.preprocess_image(img_path)
            
            # Step 2: OCR
            ocr_results = pipeline.extract_text_with_boxes(processed_img)
            
            # Step 3: Redact
            redacted_img = pipeline.redact_pii(processed_img, ocr_results)
            
            # Save Output
            output_filename = f"redacted_sample_{i}.jpg"
            cv2.imwrite(output_filename, redacted_img)
            print(f"Saved redacted image to {output_filename}")
            
        except Exception as e:
            print(f"Failed to process image {i}: {e}")

if __name__ == "__main__":
    process_samples()
