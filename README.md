# Document Scanner Application

A Flask-based web application for document scanning and OCR (Optical Character Recognition) that allows users to upload images of documents, detect document boundaries, and extract text using either Tesseract OCR or Qwen2-VL-2B model.

## Features

- Document boundary detection
- Perspective transformation
- OCR text extraction using multiple models:
  - Tesseract OCR
  - Qwen2-VL-2B model
- Web-based interface
- Session-based model management
- Image preprocessing and enhancement

## Prerequisites

- Python 3.7 or higher
- Tesseract OCR installed on your system
- CUDA-compatible GPU (recommended for Qwen2-VL-2B model)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd DocScanner
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements_app.txt
```

4. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract OCR website](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

5. Download the Qwen2-VL-2B model (if you plan to use it):
   - The model should be placed in the `Qwen2-VL-2B-OCR-fp16` directory

## Project Structure

```
DocScanner/
├── main.py              # Main Flask application
├── predictions.py       # OCR prediction logic
├── utils.py            # Utility functions
├── settings.py         # Application settings
├── requirements_app.txt # Python dependencies
├── templates/          # HTML templates
├── static/            # Static files (CSS, JS, images)
├── output/            # Output directory for processed images
└── Qwen2-VL-2B-OCR-fp16/  # Qwen2 model directory
```

## Running the Application

1. Start the Flask application:
```bash
python main.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Upload an image of a document using the web interface
2. The application will automatically detect document boundaries
3. Adjust the corner points if needed
4. Choose your preferred OCR model (Tesseract or Qwen2)
5. Click "Transform" to process the image
6. View the extracted text in the results section

## Notes

- For best results, ensure the document is well-lit and the image is clear
- The Qwen2 model requires significant GPU memory and may take longer to process
- The application supports various image formats (JPG, PNG, etc.)

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are properly installed
2. Verify Tesseract OCR is installed and accessible in your system PATH
3. Check if the Qwen2 model is properly downloaded and placed in the correct directory
4. Make sure you have sufficient disk space and memory

## License

[Your License Information]

## Acknowledgments

- Flask for the web framework
- OpenCV for image processing
- Tesseract OCR for text recognition
- Qwen2-VL-2B for advanced OCR capabilities 