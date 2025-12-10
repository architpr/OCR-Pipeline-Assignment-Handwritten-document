import streamlit as st
import cv2
import numpy as np
from handwritten_ocr import HandwrittenOCR
import tempfile
import os

# Page Config
st.set_page_config(page_title="Handwritten Medical OCR", layout="wide")

# Title and Description
st.title("Handwritten Medical Note OCR (TrOCR)")
st.markdown("""
Upload a handwritten medical note image. The system will:
1.  **Preprocess** (Grayscale, Blur, Otsu, Deskew)
2.  **Detect Text** (EasyOCR)
3.  **Recognize Handwriting** 
""")

# Cache the loader so we don't reload models on every interaction
@st.cache_resource
def load_pipeline():
    return HandwrittenOCR()

try:
    with st.spinner("Loading TrOCR Model (This may take a while )..."):
        pipeline = load_pipeline()
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("Error loading image.")
    else:
        # Initialize rotation state
        if 'rotation' not in st.session_state:
            st.session_state.rotation = 0
            
        # Rotation Controls
        st.sidebar.subheader("Image Controls")
        col_rot1, col_rot2 = st.sidebar.columns(2)
        with col_rot1:
            if st.button("Rotate Left ↺"):
                st.session_state.rotation = (st.session_state.rotation - 90) % 360
        with col_rot2:
            if st.button("Rotate Right ↻"):
                st.session_state.rotation = (st.session_state.rotation + 90) % 360

        # Apply Rotation
        if st.session_state.rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif st.session_state.rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif st.session_state.rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image (Rotated)")
            # Convert BGR to RGB for display
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Process Button
        if st.button("Extract Text"):
            with st.spinner("Processing..."):
                try:
                    # We need to save the uploaded image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_filename = tmp_file.name
                        cv2.imwrite(tmp_filename, image)
                    
                    # Step 1: Preprocess
                    processed_img = pipeline.preprocess_image(tmp_filename)
                    
                    # Step 2: OCR
                    ocr_results = pipeline.extract_text_with_boxes(processed_img)

                    # Get Annotated Image
                    annotated_img = pipeline.get_annotated_image(processed_img, ocr_results)
                    
                    # Cleanup temp file
                    os.remove(tmp_filename)
                    
                    # Display Text Results
                    st.divider()
                    st.subheader("Extracted Text")
                    
                    # Construct full text from results
                    full_text = "\n".join([text for _, text, _ in ocr_results])
                    st.text_area("Content", full_text, height=300)

                    # Display Visual Results
                    st.divider()
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.subheader("Preprocessed Input")
                        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                    with col_result2:
                        st.subheader("Annotated Output")
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                    # Download Button for Text
                    st.download_button(
                        label="Download Extracted Text",
                        data=full_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
