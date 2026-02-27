# AI-Powered Business Card Entity Extraction Web App

> A production-grade, multi-model AI system for intelligent business card scanning, OCR, and structured entity extraction — built with Flask, OpenCV, SpaCy NER, Qwen2-VL-2B, and Azure Form Recognizer.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Architecture](#architecture)
   - [High-Level System Architecture](#high-level-system-architecture)
   - [Component-Level Architecture](#component-level-architecture)
   - [Data Flow Diagram](#data-flow-diagram)
4. [Key Components](#key-components)
5. [Approaches Explored & Design Decisions](#approaches-explored--design-decisions)
6. [Key Technical Concepts](#key-technical-concepts)
7. [Key Achievements & Performance Highlights](#key-achievements--performance-highlights)
8. [Tech Stack & Justifications](#tech-stack--justifications)
9. [Repository Structure](#repository-structure)
10. [Setup & Installation](#setup--installation)
11. [Usage](#usage)

---

## Problem Statement

Business cards remain one of the most widely used tools for professional networking. However, manually transcribing contact information from physical or photographed business cards into digital systems is:

- **Time-consuming** — entering names, emails, phone numbers, and job titles one field at a time.
- **Error-prone** — misreading handwriting, abbreviations, or unconventional layouts leads to data corruption.
- **Unscalable** — enterprises receiving hundreds of cards at events have no reliable automated pipeline.
- **Inconsistent** — different card designs, fonts, languages, and orientations break simple rule-based extractors.

There is a clear need for an intelligent, multi-model system that can reliably scan business card images, correct skew and perspective distortions, extract raw text with high fidelity, and then identify and structure entities like Name, Organization, Designation, Phone, Email, and Website — even in challenging real-world conditions.

---

## Solution Overview

This project delivers a **Flask-based web application** that combines classic computer vision techniques with state-of-the-art AI models to automate end-to-end business card digitization:

1. **Image Ingestion** — Users upload a business card image through the web UI.
2. **Document Boundary Detection** — OpenCV's contour detection and Canny edge detection automatically locate the four corners of the card, even against complex backgrounds.
3. **Perspective Correction** — A four-point perspective transform rectifies skew and produces a clean, top-down view of the card.
4. **Image Enhancement** — Adaptive brightness and contrast adjustment improves OCR accuracy on low-quality inputs.
5. **Multi-Model OCR & NER** — Three interchangeable inference backends are offered:
   - **Pytesseract + SpaCy NER** — A classical OCR pipeline feeding into a custom-trained Named Entity Recognition model.
   - **Qwen2-VL-2B** — A 2-billion-parameter vision-language model performing direct visual entity extraction.
   - **Azure Form Recognizer** — A cloud-based prebuilt business card model for enterprise-grade accuracy.
6. **Structured Output** — All backends return a unified dictionary of entities: `NAME`, `ORG`, `DES`, `PHONE`, `EMAIL`, `WEB`.

The system is designed to be **modular**, **extensible**, and **deployable on GPU or CPU** hardware, making it suitable for both research prototyping and production use.

---

## Architecture

### High-Level System Architecture

```mermaid
flowchart TD
    classDef user     fill:#4f46e5,stroke:#3730a3,color:#fff,font-weight:bold
    classDef flask    fill:#0ea5e9,stroke:#0284c7,color:#fff,font-weight:bold
    classDef cv       fill:#10b981,stroke:#059669,color:#fff,font-weight:bold
    classDef engine   fill:#f59e0b,stroke:#d97706,color:#000,font-weight:bold
    classDef backend  fill:#6366f1,stroke:#4f46e5,color:#fff,font-weight:bold
    classDef output   fill:#ec4899,stroke:#db2777,color:#fff,font-weight:bold

    A(["👤 User · Web Browser\nUpload Image + Select OCR Model"]):::user

    subgraph FLASK["🌐 Flask Web Server · main.py"]
        direction LR
        R1["/ \nUpload + Detect"]:::flask
        R2["/transform\nPerspective Transform"]:::flask
        R3["/prediction\nOCR + NER"]:::flask
        R1 --> R2 --> R3
    end

    A --> R1

    R1 --> CV1["🔍 OpenCV · DocumentScan\nCanny · Contour · Polygon Detection"]:::cv
    R2 --> CV2["🔄 OpenCV · imutils\nfour_point_transform\nbrightness / contrast"]:::cv

    R3 --> INF{{"⚙️ Selected\nBackend"}}:::engine

    INF --> B1["📝 Pytesseract + SpaCy NER\nOCR → BIO Tags → Entities"]:::backend
    INF --> B2["🤖 Qwen2-VL-2B\nVision-Language Model\nFP16 · device_map=auto"]:::backend
    INF --> B3["☁️ Azure Form Recognizer\nCloud API · v2.1"]:::backend

    B1 & B2 & B3 --> OUT[["📋 Structured Output\nNAME · ORG · DES · PHONE · EMAIL · WEB"]]:::output
```

---

### Component-Level Architecture

```mermaid
graph LR
    classDef app      fill:#4f46e5,stroke:#3730a3,color:#fff,font-weight:bold
    classDef service  fill:#0ea5e9,stroke:#0284c7,color:#fff,font-weight:bold
    classDef util     fill:#10b981,stroke:#059669,color:#fff,font-weight:bold
    classDef config   fill:#64748b,stroke:#475569,color:#fff,font-weight:bold
    classDef model    fill:#f59e0b,stroke:#d97706,color:#000,font-weight:bold
    classDef template fill:#ec4899,stroke:#db2777,color:#fff,font-weight:bold

    MAIN["🏠 main.py\nFlask App · Routes · Session"]:::app

    subgraph SVC["services/"]
        PRED["predictions.py\nPytesseract + SpaCy NER"]:::service
        QWEN["qwenform.py\nQwen2-VL-2B"]:::service
        AZURE["azureform.py\nAzure Form Recognizer"]:::service
    end

    subgraph UTL["utils/"]
        U["utils.py\nDocumentScan Pipeline"]:::util
    end

    subgraph CFG["config/"]
        S["settings.py\nPath Constants"]:::config
    end

    subgraph MDL["models/"]
        MB["model-best/\nSpaCy NER (best ckpt)"]:::model
        ML["model-last/\nSpaCy NER (last ckpt)"]:::model
    end

    subgraph TPL["templates/"]
        T1["scanner.html"]:::template
        T2["predictions.html"]:::template
        T3["qwen_prediction.html"]:::template
        T4["azure_prediction.html"]:::template
    end

    MAIN --> PRED & QWEN & AZURE
    MAIN --> U
    MAIN --> S
    U --> S
    PRED --> MB & ML
    PRED --> S
    QWEN --> S
    AZURE --> S
    MAIN --> T1 & T2 & T3 & T4
```

---

### Data Flow Diagram

```mermaid
flowchart TD
    classDef io       fill:#0ea5e9,stroke:#0284c7,color:#fff,font-weight:bold
    classDef proc     fill:#10b981,stroke:#059669,color:#fff,font-weight:bold
    classDef decision fill:#f59e0b,stroke:#d97706,color:#000,font-weight:bold
    classDef modelA   fill:#6366f1,stroke:#4f46e5,color:#fff,font-weight:bold
    classDef modelB   fill:#8b5cf6,stroke:#7c3aed,color:#fff,font-weight:bold
    classDef modelC   fill:#0ea5e9,stroke:#0284c7,color:#fff,font-weight:bold
    classDef output   fill:#ec4899,stroke:#db2777,color:#fff,font-weight:bold
    classDef file     fill:#64748b,stroke:#475569,color:#fff

    A(["📷 User Uploads Image"]):::io
    B["save_upload_image()\n→ static/media/upload.jpg"]:::file
    C["DocumentScan.document_scanner()\nResize to 500px · Detail Enhance\nGrayscale · Gaussian Blur 5×5\nCanny Edge 75/200 · Morph Close\nContour → largest 4-corner polygon"]:::proc
    D(["🖱️ Browser: User drags corner points\nto fine-tune card boundary"]):::io
    E["POST /transform\ncalibrate_to_original_size()\nfour_point_transform()\napply_brightness_contrast(+40, +60)\n→ static/media/magic_color.jpg"]:::proc
    F{{"GET /prediction\nOCR model from session"}}:::decision

    G["📝 Pytesseract + SpaCy NER\nimage_to_data() word bboxes\nText cleaning · SpaCy NER inference\nBIO tag decode · entity grouping\nRegex normalization per field type\n→ bounding_box.jpg overlay"]:::modelA
    H["🤖 Qwen2-VL-2B\nLoad FP16 · device_map=auto\nResize 640×640\nChat template + JSON schema prompt\nGenerate max_new_tokens=512 greedy\nJSON extract + validate"]:::modelB
    I["☁️ Azure Form Recognizer\nPOST /prebuilt/businessCard/analyze\nExponential backoff polling\nField extraction + empty-field strip"]:::modelC

    J[["📋 Structured Entities\nNAME · ORG · DES · PHONE · EMAIL · WEB\n→ Rendered HTML template"]]:::output

    A --> B --> C --> D --> E --> F
    F -->|pytesseract| G
    F -->|qwen2| H
    F -->|azure| I
    G & H & I --> J
```

---

## Key Components

### `utils/utils.py` — Document Preprocessing Pipeline

The `DocumentScan` class implements the full computer vision pre-processing pipeline:

| Step | Method / Technique | Purpose |
|---|---|---|
| Resize | `cv2.resize` to 500px width | Normalize input for consistent edge detection |
| Detail Enhancement | `cv2.detailEnhance` (sigma_s=20, sigma_r=0.15) | Sharpen edges while preserving structure |
| Grayscale + Blur | `cvtColor` + `GaussianBlur` (5×5) | Reduce noise before edge detection |
| Edge Detection | `cv2.Canny` (75, 200) | Detect card boundaries |
| Morphological Ops | Dilation + `MORPH_CLOSE` (5×5 kernel) | Fill gaps in detected edges |
| Contour Analysis | `findContours` + `approxPolyDP` | Find the largest 4-vertex polygon (the card) |
| Perspective Transform | `four_point_transform` (imutils) | Correct skew and produce top-down view |
| Enhancement | `addWeighted` brightness/contrast | Improve OCR legibility of warped image |

---

### `services/predictions.py` — Pytesseract + SpaCy NER Backend

This backend implements a classical NLP pipeline:

1. **OCR with Pytesseract** — Extracts word-level bounding boxes and confidence scores via `image_to_data`.
2. **Text Cleaning** — Strips whitespace and punctuation to normalize tokens before NER.
3. **SpaCy NER Inference** — A custom-trained SpaCy NER model loaded from `models/model-best/` tags tokens with `B-`/`I-` prefixed labels: `NAME`, `ORG`, `DES`, `PHONE`, `EMAIL`, `WEB`.
4. **BIO Decoding** — The `groupgen` class groups consecutive tokens of the same entity type.
5. **Field-Specific Parsing** — The `parser()` function applies regex normalization per entity type (e.g., strip non-digits from phone, normalize URL characters for web).
6. **Bounding Box Overlay** — Draws green rectangles and magenta labels on the original image for each detected entity group.

---

### `services/qwenform.py` — Qwen2-VL-2B Vision-Language Backend

This backend leverages a modern vision-language model for direct image-to-JSON extraction:

- **Model**: `Qwen2-VL-2B-OCR-fp16` — a 2B-parameter vision encoder + language decoder fine-tuned for OCR tasks.
- **Loading**: `AutoModelForVision2Seq` with `torch_dtype=float16` on CUDA, or `float32` on CPU. `device_map="auto"` enables automatic GPU/CPU allocation.
- **Prompt Engineering**: A structured JSON schema is injected into the chat template, instructing the model to extract exactly the 6 entity fields and respond only with valid JSON.
- **Inference**: Greedy decoding (`do_sample=False`, `num_beams=1`) for deterministic, fast extraction at `max_new_tokens=512`.
- **Post-processing**: JSON is extracted from the raw generation output using string search for `{...}` delimiters, followed by `json.loads` validation and missing-key backfilling.
- **Lazy Loading**: The model is loaded on-demand via the `/load_qwen_model` endpoint to avoid startup latency when using other backends.

---

### `services/azureform.py` — Azure Form Recognizer Backend

This backend delegates to Microsoft's prebuilt business card model:

- **API**: Azure Form Recognizer v2.1 `/prebuilt/businessCard/analyze` endpoint.
- **Async Polling**: Submits the image for analysis, then polls the `operation-location` URL with exponential backoff (initial wait 2s, multiplier 1.5×, up to 5 retries) to handle variable cloud processing times.
- **Field Mapping**: Extracts `ContactNames` (firstName/lastName), `JobTitles`, `CompanyNames`, `Addresses`, `OtherPhones`, `Faxes`, `Emails`, and `Websites` from the Form Recognizer response schema.
- **Robustness**: Empty fields are stripped from the output; all extraction is wrapped in exception handlers to surface API errors cleanly.
- **Configuration**: Endpoint URL and API key are loaded from environment variables via `python-dotenv`, keeping credentials out of source code.

---

### `main.py` — Flask Application & Orchestration

The Flask application serves as the orchestration layer:

| Route | Method | Description |
|---|---|---|
| `/` | GET/POST | Upload image, run document boundary detection, return corner coordinates as JSON to the browser |
| `/transform` | POST | Receive adjusted corner points from UI, apply perspective transform and enhancement |
| `/prediction` | GET | Run the session-selected OCR backend and render results |
| `/load_qwen_model` | POST | Lazy-load the Qwen2 model into GPU/CPU memory |
| `/about` | GET | About page |

Session management (`flask.session`) persists the OCR model choice across the upload → transform → predict flow, enabling a stateful multi-step user experience.

---

## Approaches Explored & Design Decisions

### 1. Multi-Backend Architecture
Rather than committing to a single OCR strategy, the system was designed with **three interchangeable backends** to enable direct quality comparison across different technology paradigms:

| Backend | Approach | Strengths | Limitations |
|---|---|---|---|
| Pytesseract + SpaCy | Classical OCR + custom NER | Fast, CPU-friendly, fully offline, interpretable bounding boxes | Depends on OCR quality; fails on heavily stylized fonts |
| Qwen2-VL-2B | Vision-language model | Understands visual context, robust to unusual layouts, no separate OCR step | Requires GPU for practical speed; heavier dependency footprint |
| Azure Form Recognizer | Cloud prebuilt model | Highest accuracy on standard cards, handles multiple languages | Requires internet + API key; incurs per-request cloud cost |

### 2. Custom SpaCy NER Model Training
The SpaCy model was trained specifically on business card text rather than using a general-purpose NER model. This is a key design decision because:
- Business card entities (`DES`, `WEB`) are not present in standard NER tagsets (PERSON, ORG, etc.).
- Domain-specific training data yields far higher precision and recall on this narrow extraction task.
- The model uses BIO (Beginning-Inside-Outside) tagging, which handles multi-token entities like full names and organization names correctly.

### 3. Perspective Transform Before OCR
A critical pre-processing step often overlooked in naive OCR pipelines: **correcting document skew**. Feeding a rotated or trapezoidally-distorted card image directly to an OCR engine significantly degrades output quality. By detecting the card boundary and applying a homographic transformation first, the system normalizes the input into a clean, rectangular, uniformly lit image.

### 4. Interactive Corner Adjustment
Rather than relying solely on automated boundary detection, the UI allows users to **drag and adjust the detected corner points** before applying the transform. This hybrid human-in-the-loop design handles cases where automatic contour detection is imperfect (e.g., cards with low contrast against the background).

### 5. Lazy Model Loading for Qwen2
Loading a 2B-parameter model takes significant time and memory. The Qwen2 model is **not loaded at application startup** — instead, it is loaded on-demand via an explicit `/load_qwen_model` endpoint. This avoids startup delays and unnecessary memory consumption when users are running the Pytesseract or Azure backends.

### 6. Greedy Decoding for Reproducibility
The Qwen2 backend uses greedy decoding (`do_sample=False`, `num_beams=1`, `early_stopping=True`) rather than sampling. For structured data extraction tasks, deterministic outputs are preferable to sampled ones, as they improve reproducibility and make debugging easier.

### 7. Exponential Backoff for Azure Polling
Azure Form Recognizer processes requests asynchronously. A naive implementation using fixed-interval polling wastes API calls during fast completions and hammers the endpoint during slow ones. The exponential backoff strategy (2s → 3s → 4.5s → 6.75s → 10.1s) balances responsiveness with API efficiency.

---

## Key Technical Concepts

### Computer Vision
- **Canny Edge Detection** — Gradient-based multi-threshold edge detector for finding card boundaries.
- **Morphological Transforms** — Dilation and closing operations to strengthen and close edge gaps.
- **Contour Analysis + Polygon Approximation** — `findContours` + `approxPolyDP` to identify the dominant four-vertex quadrilateral in the image.
- **Homographic Perspective Transform** — `four_point_transform` maps the detected quadrilateral to a rectangle, correcting camera perspective.
- **Adaptive Image Enhancement** — Linear brightness/contrast adjustment using a weighted blend formula for improved OCR conditions.

### Natural Language Processing
- **BIO Tagging Schema** — Each token is tagged as `B-{ENTITY}` (beginning), `I-{ENTITY}` (inside), or `O` (outside) for sequence labeling.
- **Named Entity Recognition (NER)** — SpaCy pipeline performing token classification over OCR-extracted text.
- **Token Grouping** — The `groupgen` class tracks label transitions to correctly group multi-token entities into single values.
- **Field-Level Normalization** — Regex-based post-processing per entity type (phone numbers stripped to digits, emails lowercased, URLs cleaned).

### Vision-Language Models
- **Vision Encoder + Language Decoder** — Qwen2-VL-2B encodes the image using a vision transformer and feeds visual tokens alongside text tokens to a causal language model.
- **Chat Template Prompting** — Structured system + user messages with an explicit JSON schema in the prompt guide the model toward structured extraction outputs.
- **Mixed Precision Inference** — FP16 inference on CUDA reduces memory footprint by ~50% versus FP32 with minimal accuracy impact.
- **Device Map Auto** — HuggingFace's automatic device placement splits model layers across available GPU(s) and CPU RAM.

### System Design
- **Session-Based State** — Flask sessions carry OCR model selection across the multi-step request flow without requiring a database.
- **Modular Service Layer** — Each OCR backend is a self-contained module with a standard interface, making it straightforward to add new backends.
- **Graceful Error Handling** — All processing paths return structured error dictionaries rather than raising unhandled exceptions, ensuring the UI always receives a renderable response.

---

## Key Achievements & Performance Highlights

- **Three fully integrated inference backends** in a single unified web application, enabling direct A/B comparison of different AI approaches.
- **End-to-end pipeline** from raw camera image to structured JSON entity dictionary with no manual data entry.
- **Robust boundary detection** handles cards photographed at angles, in cluttered scenes, and with variable lighting conditions.
- **Custom SpaCy NER model** trained specifically for business card domain entity types not found in standard NER tagsets.
- **GPU-accelerated vision-language inference** with FP16 mixed precision, reducing Qwen2 memory footprint and improving throughput.
- **Cloud-native integration** with Azure Form Recognizer for enterprise-grade accuracy without maintaining a heavy local model.
- **Interactive UI** with drag-adjustable corner points, providing a human-in-the-loop fallback for edge cases where automatic detection is imperfect.
- **Lazy model loading** prevents unnecessary startup overhead and GPU memory allocation when using lighter backends.
- **Stateless-friendly architecture** — the Flask session layer provides just enough state for multi-step UX without complex backend persistence.

---

## Tech Stack & Justifications

| Technology | Role | Justification |
|---|---|---|
| **Python 3.7+** | Core language | Rich ecosystem for CV, ML, and web development |
| **Flask 3.0** | Web framework | Lightweight, minimal boilerplate; ideal for ML serving APIs |
| **OpenCV 4.9** | Image processing | Industry-standard CV library with full contour/transform support |
| **imutils** | Perspective transform | Clean `four_point_transform` implementation built on OpenCV |
| **Pytesseract 0.3.10** | OCR engine wrapper | Python wrapper for Tesseract, the most widely used open-source OCR |
| **SpaCy 3.7** | NLP + NER | Production-grade NLP library with fast, trainable NER pipelines |
| **pandas / NumPy** | Data manipulation | Efficient tabular processing of Pytesseract's word-level output |
| **Transformers (HuggingFace)** | Vision-language model | Standard library for loading and running Qwen2-VL-2B |
| **PyTorch** | Deep learning backend | CUDA-enabled tensor operations for Qwen2 inference |
| **Pillow** | Image I/O for PyTorch | PIL format required by HuggingFace vision processors |
| **Azure Form Recognizer** | Cloud OCR API | Prebuilt business card model for high-accuracy cloud inference |
| **python-dotenv** | Secret management | Loads Azure credentials from `.env`, keeping secrets out of source code |
| **Jinja2** | HTML templating | Flask-native; enables dynamic result rendering with minimal JS |
| **Werkzeug** | File upload security | `secure_filename` sanitizes uploaded file names |

---

## Repository Structure

```
AI-Powered-Business-Card-Entity-Extraction-Web-App/
│
├── main.py                     # Flask application entry point; all routes and orchestration
│
├── config/
│   └── settings.py             # Path constants (BASE_DIR, MEDIA_DIR, SAVE_DIR)
│
├── utils/
│   └── utils.py                # DocumentScan class: full image preprocessing pipeline
│                               #   - resizer(), apply_brightness_contrast()
│                               #   - document_scanner(), calibrate_to_original_size()
│                               # save_upload_image(), array_to_json_format()
│
├── services/
│   ├── predictions.py          # Pytesseract + SpaCy NER backend
│   │                           #   - cleanText(), parser(), groupgen, getPredictions()
│   │                           #   - extract_json_response() (utility)
│   ├── qwenform.py             # Qwen2-VL-2B vision-language backend
│   │                           #   - load_qwen_model(), process_document()
│   └── azureform.py            # Azure Form Recognizer backend
│                               #   - process_business_card(), extract_business_card_data()
│
├── models/
│   ├── model-best/             # Best-performing SpaCy NER model checkpoint
│   └── model-last/             # Final epoch SpaCy NER model checkpoint
│
├── templates/
│   ├── scanner.html            # Main upload + interactive corner-point adjustment UI
│   ├── predictions.html        # Pytesseract/SpaCy results with annotated image
│   ├── qwen_prediction.html    # Qwen2-VL-2B results display
│   ├── azure_prediction.html   # Azure Form Recognizer results display
│   ├── about.html              # About page
│   └── index.html              # Landing page
│
├── static/
│   ├── css/                    # Stylesheets
│   ├── js/                     # JavaScript (corner-point drag UI, AJAX model loading)
│   └── images/                 # Static image assets
│
├── test/                       # Sample business card images for testing
│   ├── 001.jpg
│   ├── 015.jpg
│   ├── 03.jpg
│   ├── 033.jpg
│   ├── test1.jpg
│   ├── test2.jpg
│   └── test3.jpg
│
└── requirements_app.txt        # All Python dependencies with pinned versions
```

---

## Setup & Installation

### Prerequisites

- Python 3.7 or higher
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system
- CUDA-compatible GPU *(recommended for Qwen2-VL-2B; CPU inference is supported but slow)*
- Azure Form Recognizer resource *(only required for the Azure backend)*

### 1. Clone the Repository

```bash
git clone https://github.com/HarshaVardhanMannem/AI-Powered-Business-Card-Entity-Extraction-Web-App.git
cd AI-Powered-Business-Card-Entity-Extraction-Web-App
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Python Dependencies

```bash
pip install -r requirements_app.txt
```

### 4. Install Tesseract OCR

| OS | Command |
|---|---|
| **Ubuntu/Debian** | `sudo apt-get install tesseract-ocr` |
| **macOS** | `brew install tesseract` |
| **Windows** | Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH |

### 5. (Optional) Download the Qwen2-VL-2B Model

If you plan to use the Qwen2 backend, download the fine-tuned model and place it at:

```
models/Qwen2-VL-2B-OCR-fp16/
```

The model can be downloaded from HuggingFace or a compatible model repository.

### 6. (Optional) Configure Azure Form Recognizer

Create a `.env` file in the project root:

```env
AZURE_FORM_RECOGNIZER_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_FORM_RECOGNIZER_KEY=<your-api-key>
```

### 7. Run the Application

```bash
python main.py
```

Navigate to `http://localhost:5000` in your browser.

---

## Usage

### Step-by-Step Workflow

1. **Open the web application** at `http://localhost:5000`.

2. **Select an OCR model** from the dropdown:
   - `Pytesseract + SpaCy` — Fast, offline, CPU-friendly.
   - `Qwen2-VL-2B` — High accuracy, requires GPU for practical speed. Click **"Load Qwen Model"** before uploading.
   - `Azure Form Recognizer` — Cloud-based, highest accuracy; requires configured Azure credentials.

3. **Upload a business card image** (JPG, PNG supported).

4. **Review detected corner points** — The system automatically highlights the four corners of the card. Drag any point to adjust if needed.

5. **Click "Transform"** — The image is perspective-corrected and enhanced.

6. **View extracted entities** — The results page displays:
   - Annotated image with bounding boxes (Pytesseract mode)
   - Structured entity table: Name, Organization, Designation, Phone, Email, Website

### Tips for Best Results

- Use well-lit images with the card as the dominant subject.
- Avoid heavy shadows or extreme angles (the interactive corner adjustment can compensate for moderate skew).
- For heavily stylized cards with unusual fonts, prefer the Qwen2 or Azure backends over Pytesseract.
- For batch processing at scale, prefer the Azure backend which is designed for production workloads.

---

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/) — Web framework
- [OpenCV](https://opencv.org/) — Computer vision and image processing
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Open-source OCR engine
- [SpaCy](https://spacy.io/) — Industrial-strength NLP and NER
- [Qwen2-VL](https://huggingface.co/Qwen) — Vision-language model for direct visual entity extraction
- [Azure Form Recognizer](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence) — Cloud-based document intelligence
- [imutils](https://github.com/jrosebr1/imutils) — Convenience functions for OpenCV, including `four_point_transform`
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — Model loading and inference framework 