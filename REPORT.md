# CS419 - Assignment 02: Image Retrieval with Relevance Feedback

## Student Information
- **Name:** Nguyen Huu Hoang Minh
- **Student ID:** 22125058
- **Course:** CS419 - Introduction to Information Retrieval
- **University:** VNU-HCM University of Science

---

## 1. Introduction

This project implements a **Content-Based Image Retrieval (CBIR)** system with **Relevance Feedback** using the **Rocchio Algorithm**. The system is designed for anime image datasets and features a modern web-based GUI built with Gradio.

### 1.1 Objectives
- Implement image retrieval using deep learning features (CLIP/ResNet50)
- Apply the Rocchio relevance feedback algorithm to refine search results
- Build an intuitive GUI for interactive image search

### 1.2 Features
- Support for two feature extraction models: **CLIP** (recommended) and **ResNet50**
- Combined features: Deep learning (85%) + Color histogram (15%)
- Interactive relevance feedback with visual marking
- Progress tracking with ETA during feature extraction
- Feature caching for fast subsequent loads

---

## 2. System Architecture

### 2.1 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio Web GUI                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Setup   │  │  Search  │  │  Browse  │  │  About   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer (app.py)                   │
│  • initialize_system()  • search_by_image()  • apply_feedback() │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Core Retrieval (image_retrieval.py)              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │FeatureExtractor │  │  ImageDatabase  │  │RelevanceFeedback│  │
│  │  • CLIP/ResNet  │  │  • Features DB  │  │    Retrieval    │  │
│  │  • Color Hist   │  │  • Caching      │  │  • Rocchio Alg  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐   │
│  │  Anime Images   │  │  Feature Cache (features_cache.pkl) │   │
│  │  (82,975 imgs)  │  │  • CLIP: 345 MB                     │   │
│  └─────────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

#### 2.2.1 FeatureExtractor Class
Extracts visual features from images using two methods:

| Component | Description | Dimension |
|-----------|-------------|-----------|
| **CLIP ViT-B-32** | Semantic features from OpenAI's CLIP model | 512 |
| **ResNet50** | CNN features from ImageNet-pretrained model | 2048 |
| **Color Histogram** | HSV color distribution (8×8×8 bins) | 512 |

**Feature Combination:**
```
combined_features = CNN_features × 0.85 + Color_features × 0.15
```

#### 2.2.2 ImageDatabase Class
Manages the image collection and precomputed features:
- Scans image folders recursively
- Extracts and caches features for fast loading
- Supports multiple image formats (JPG, PNG, WebP, GIF)

#### 2.2.3 RelevanceFeedbackRetrieval Class
Implements the Rocchio algorithm for query refinement:

**Rocchio Formula:**
```
q_new = α × q_old + β × (1/|Dr|) × Σ(d ∈ Dr) - γ × (1/|Dnr|) × Σ(d ∈ Dnr)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| α (alpha) | 1.0 | Weight for original query |
| β (beta) | 0.75 | Weight for relevant documents |
| γ (gamma) | 0.25 | Weight for non-relevant documents |

---

## 3. Algorithm Details

### 3.1 Feature Extraction Pipeline

```
Input Image
     │
     ▼
┌─────────────┐
│ Resize to   │
│ 224 × 224   │
└─────────────┘
     │
     ├──────────────────┐
     ▼                  ▼
┌─────────────┐   ┌─────────────┐
│    CLIP     │   │   Color     │
│  Encoding   │   │ Histogram   │
│  (512-dim)  │   │  (512-dim)  │
└─────────────┘   └─────────────┘
     │                  │
     │    ┌─────────────┘
     ▼    ▼
┌─────────────┐
│  Weighted   │
│ Concatenate │
│ (1024-dim)  │
└─────────────┘
     │
     ▼
┌─────────────┐
│ L2 Normalize│
└─────────────┘
     │
     ▼
Feature Vector
```

### 3.2 Similarity Search

**Cosine Similarity:**
```
similarity(q, d) = (q · d) / (||q|| × ||d||)
```

The system ranks all database images by their cosine similarity to the query vector and returns the top-K results.

### 3.3 Relevance Feedback Process

```
┌─────────────────────────────────────────────────────────────┐
│                    User Workflow                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  1. Upload Query      │
              │     Image             │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  2. Initial Search    │
              │  (Extract features,   │
              │   compute similarity) │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  3. Display Top-K     │
              │     Results           │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  4. User Marks        │
              │  ✅ Relevant          │
              │  ❌ Non-Relevant      │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  5. Apply Rocchio     │
              │     Algorithm         │
              │  q_new = αq + βDr     │
              │         - γDnr        │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  6. Re-rank with      │
              │     New Query         │
              └───────────────────────┘
                          │
                          ▼
                   Repeat 3-6
              until user satisfied
```

---

## 4. Implementation

### 4.1 Project Structure

```
anime_image_retrieval_system/
├── app.py                 # Gradio GUI application
├── image_retrieval.py     # Core retrieval algorithms
├── requirements.txt       # Python dependencies
├── README.md              # User documentation
├── REPORT.md              # This report
├── anime_images/          # Dataset folder
│   ├── Attack on Titan/
│   ├── Naruto/
│   └── ... (231 anime folders)
└── venv/                  # Virtual environment
```

### 4.2 Key Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Programming language |
| PyTorch | 2.0+ | Deep learning framework |
| OpenCLIP | 2.20+ | CLIP model implementation |
| Gradio | 6.0+ | Web GUI framework |
| OpenCV | 4.5+ | Image processing |
| NumPy | 1.21+ | Numerical computing |

### 4.3 GUI Design

The application has 4 tabs:

1. **Setup Tab**: Initialize system, select model, adjust Rocchio parameters
2. **Search Tab**: Upload query, view results, mark relevance, apply feedback
3. **Browse Tab**: Search by database index (for testing)
4. **About Tab**: System information and algorithm details

---

## 5. Results and Evaluation

### 5.1 Dataset
- **Source:** Anime Images Dataset (Kaggle)
- **Size:** 82,975 images
- **Categories:** 231 anime series
- **Format:** JPG, PNG

### 5.2 Performance

| Metric | CLIP | ResNet50 |
|--------|------|----------|
| Feature extraction speed | ~12 img/s | ~15 img/s |
| Feature dimension | 1024 | 2560 |
| Cache file size | 345 MB | 854 MB |
| Semantic understanding | Good | Poor |
| Character recognition | Good | Poor |

### 5.3 Observations

1. **CLIP vs ResNet50**: CLIP significantly outperforms ResNet50 for anime character retrieval due to its training on diverse image-text pairs.

2. **Relevance Feedback**: After 2-3 iterations of feedback, search results typically improve substantially when users mark both relevant and non-relevant images.

3. **Color Histogram**: Adding color features (15% weight) helps distinguish anime with similar art styles but different color palettes.

---

## 6. How to Run

### 6.1 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Running the Application

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### 6.3 Usage Steps

1. Go to **Setup** tab → Click **Initialize System**
2. Wait for feature extraction (first run) or cache loading
3. Go to **Search** tab → Upload a query image
4. Click **Search** to get initial results
5. Mark images as **Relevant** or **Non-Relevant**
6. Click **Apply Relevance Feedback**
7. Repeat steps 5-6 until satisfied

---

## 7. Conclusion

This project successfully implements a content-based image retrieval system with relevance feedback for anime images. Key achievements:

1. **Effective Feature Extraction**: Using CLIP provides strong semantic understanding for anime content.

2. **Interactive Feedback**: The Rocchio algorithm effectively refines search results based on user feedback.

3. **User-Friendly Interface**: The Gradio GUI makes the system accessible and easy to use.

### Future Improvements

- Add support for text-based search using CLIP's text encoder
- Implement batch processing for faster feature extraction
- Add more advanced feedback mechanisms (e.g., neural network-based)
- Support for larger datasets using approximate nearest neighbor search (FAISS)

---

## 8. References

1. Rocchio, J.J. (1971). Relevance feedback in information retrieval. *The SMART Retrieval System*, 313-323.

2. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.

3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.

4. CS419 Lecture Slides - Interactive Search with Relevance Feedback.

---

## Appendix A: Rocchio Algorithm Pseudocode

```python
def rocchio_feedback(query, relevant_docs, non_relevant_docs, alpha, beta, gamma):
    """
    Rocchio relevance feedback algorithm

    Args:
        query: Original query vector
        relevant_docs: List of relevant document vectors
        non_relevant_docs: List of non-relevant document vectors
        alpha: Weight for original query (default: 1.0)
        beta: Weight for relevant documents (default: 0.75)
        gamma: Weight for non-relevant documents (default: 0.25)

    Returns:
        new_query: Modified query vector
    """
    # Start with weighted original query
    new_query = alpha * query

    # Add centroid of relevant documents
    if len(relevant_docs) > 0:
        relevant_centroid = mean(relevant_docs)
        new_query += beta * relevant_centroid

    # Subtract centroid of non-relevant documents
    if len(non_relevant_docs) > 0:
        non_relevant_centroid = mean(non_relevant_docs)
        new_query -= gamma * non_relevant_centroid

    return new_query
```
