# Handwritten Medical Note OCR Pipeline (TrOCR Powered) 🏥✍️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)
![Model](https://img.shields.io/badge/Model-Microsoft%20TrOCR-green)
![License](https://img.shields.io/badge/License-MIT-grey)

A robust, object-oriented Optical Character Recognition (OCR) pipeline designed specifically for challenging **Handwritten Medical Notes**. This project leverages a **Hybrid Architecture** combining the detection capabilities of **EasyOCR** with the state-of-the-art recognition accuracy of **Microsoft TrOCR** (Transformer OCR).

## 🌟 Key Features

*   **Hybrid AI Pipeline**:
    *   **Detection**: Uses `EasyOCR` to accurately locate text regions on the page (bounding boxes).
    *   **Recognition**: Uses `Microsoft TrOCR` (based on Vision Encoder-Decoder Transformers) to read the handwriting within those regions with high accuracy.
*   **Advanced Preprocessing**:
    *   Adaptive Thresholding (Otsu's Binarization) to separate ink from paper.
    *   Automatic **Deskewing** to correct text tilt.
    *   Noise Reduction (Gaussian Blur).
*   **Intelligent Structure Recovery**:
    *   Algorithms to sort detected text blocks into logical lines and rows, reconstructing the document's layout.
*   **Interactive Web UI**:
    *   Built with Streamlit for easy uploading, visualization, and interaction.
    *   Manual rotation controls (Left/Right) to fix image orientation.
    *   Side-by-side comparison of original and annotated images.

## 🛠️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/architpr/OCR-Pipeline-Assignment-Handwritten-document.git
    cd OCR-Pipeline-Assignment-Handwritten-document
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will install `torch`, `transformers`, `easyocr`, `opencv-python`, and `streamlit`.*

## 🚀 Usage

### 1. Interactive Web App (Recommended)
Launch the user-friendly interface:
```bash
streamlit run app.py
```
*   Open your browser to the URL shown (usually `http://localhost:8501`).
*   Upload an image.
*   Use the sidebar controls to rotate the image if needed.
*   Click **"Extract Text"**.

### 2. Python Notebook (Demo)
Run the Jupyter Notebook for a step-by-step walkthrough of the code:
```bash
jupyter notebook handwritten_ocr_demo.ipynb
```

### 3. Usage in Scripts
You can import the `HandwrittenOCR` class into your own projects:

```python
from handwritten_ocr import HandwrittenOCR

# Initialize
ocr = HandwrittenOCR()

# Process
processed_img = ocr.preprocess_image("path/to/note.jpg")
results = ocr.extract_text_with_boxes(processed_img)

# Print Text
for _, text, conf in results:
    print(text)
```


## ☁️ Deployment (Streamlit Cloud)

This app is ready to be deployed on **Streamlit Cloud** for free!

1.  **Push to GitHub** (Already done).
2.  Go to [share.streamlit.io](https://share.streamlit.io).
3.  Click **"New App"**.
4.  Select this repository (`OCR-Pipeline-Assignment-Handwritten-document`).
5.  Set **Main file path** to `app.py`.
6.  Click **"Deploy!"**.

**Note on Resources**:
*   The system downloads large models (~1.5GB) on the first run.
*   The boot process might take 2-3 minutes.
*   If you face memory issues, reboot the app in the Streamlit Cloud dashboard.

## 📂 Project Structure

*   `app.py`: The Main Streamlit application file.
*   `handwritten_ocr.py`: Core logic containing the `HandwrittenOCR` class (Preprocessing, EasyOCR, TrOCR).
*   `handwritten_ocr_demo.ipynb`: Jupyter Notebook demonstrating the pipeline.
*   `requirements.txt`: List of Python dependencies.

## ⚠️ Requirements
*   **Memory**: The TrOCR model is approximately 1.5GB. Ensure you have sufficient RAM.
*   **GPU**: A CUDA-capable GPU is highly recommended for faster inference, but the code will fallback to CPU if not available.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
