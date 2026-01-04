# 🎨 Anime Image Retrieval with Relevance Feedback

**CS419 - Assignment 02**  
University of Science - VNU-HCM

## Overview

This system implements **Content-Based Image Retrieval (CBIR)** with **Relevance Feedback** using the **Rocchio Algorithm**. It's designed for anime image datasets but works with any image collection.

## Features

- **CNN Features (ResNet50)** - Deep semantic features for content understanding
- **Color Histogram (HSV)** - Captures anime's vibrant color patterns
- **Rocchio Relevance Feedback** - Iteratively refines search results
- **Modern Web GUI** - Built with Gradio for easy interaction

## Installation

```bash
# 1. Clone/Download the project
cd anime_retrieval

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download anime dataset from Kaggle
# https://www.kaggle.com/datasets/diraizel/anime-images-dataset
# Extract to ./anime_images/
```

## Usage

### Run the GUI Application

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### Using the System

1. **Setup Tab**: Enter dataset path and click "Initialize System"
2. **Search Tab**: Upload a query image and click "Search"
3. **Relevance Feedback**: 
   - Check images that match what you want (✅ Relevant)
   - Check images you don't want (❌ Non-Relevant)
   - Click "Apply Relevance Feedback"
4. Repeat step 3 until satisfied with results

## Algorithm Details

### Feature Extraction
- **CNN Features**: ResNet50 pretrained on ImageNet (2048-dim)
- **Color Histogram**: HSV histogram with 8 bins per channel (512-dim)
- **Combined**: 70% CNN + 30% Color weighted concatenation

### Rocchio Relevance Feedback

```
q_new = α·q + β·(1/|Dr|)·Σd∈Dr - γ·(1/|Dnr|)·Σd∈Dnr
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| α | 1.0 | Weight for original query |
| β | 0.75 | Weight for relevant documents |
| γ | 0.25 | Weight for non-relevant documents |

### Similarity Measure
- **Cosine Similarity**: `cos(θ) = (A·B)/(||A||·||B||)`

## Project Structure

```
anime_retrieval/
├── app.py              # Gradio GUI application
├── image_retrieval.py  # Core retrieval system
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── anime_images/      # Your dataset folder
    ├── image1.jpg
    ├── image2.png
    └── ...
```

## Performance Tips

- First run creates a feature cache (`features_cache.pkl`)
- Subsequent runs load from cache (much faster)
- Use GPU for faster feature extraction (auto-detected)
- For very large datasets (>10000 images), consider FAISS for faster search

## References

1. CS419 Lecture Slides - Interactive Search with Relevance Feedback
2. Rocchio, J.J. (1971). Relevance feedback in information retrieval
3. He et al. (2016). Deep Residual Learning for Image Recognition

## Author

CS419 - Introduction to Information Retrieval  
VNU-HCM University of Science

---

## Demo Video Requirements

For your submission, record a video showing:
1. System initialization with dataset
2. Initial query search
3. Multiple iterations of relevance feedback
4. Improved results after feedback
