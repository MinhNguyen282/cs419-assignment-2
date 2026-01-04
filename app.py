"""
Gradio GUI for Image Retrieval with Relevance Feedback
CS419 - Assignment 02
"""

import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
import os
from typing import List, Tuple, Optional
from image_retrieval import FeatureExtractor, ImageDatabase, RelevanceFeedbackRetrieval

# Global variables
extractor: FeatureExtractor = None
database: ImageDatabase = None
retrieval: RelevanceFeedbackRetrieval = None
current_results: List[Tuple[int, float]] = []
selected_relevant: set = set()
selected_non_relevant: set = set()


def initialize_system(dataset_path: str, model_type: str, progress=gr.Progress()):
    """Initialize the retrieval system with dataset"""
    global extractor, database, retrieval

    if not dataset_path or not os.path.exists(dataset_path):
        return "❌ Error: Please provide a valid dataset path", None

    model_name = "CLIP (Recommended)" if model_type == "clip" else "ResNet50"
    progress(0, desc=f"Loading {model_name} model...")
    extractor = FeatureExtractor(use_gpu=True, model_type=model_type)

    progress(0.1, desc="Scanning image folder...")
    database = ImageDatabase(extractor)

    # Create progress callback for detailed progress updates
    def progress_callback(current, total, message):
        if total > 0:
            # Reserve 0.1-0.9 for feature extraction (80% of progress bar)
            pct = 0.1 + (current / total) * 0.8
            progress(pct, desc=message)

    database.build_database(dataset_path, progress_callback=progress_callback)

    progress(0.95, desc="Initializing retrieval system...")
    retrieval = RelevanceFeedbackRetrieval(database, extractor)

    progress(1.0, desc="Done!")

    # Get sample images for display
    sample_paths = database.image_paths[:8]
    sample_images = [(Image.open(p).resize((150, 150)), f"Image {i}") for i, p in enumerate(sample_paths)]

    model_info = "CLIP (better for anime characters)" if model_type == "clip" else "ResNet50"
    return f"✅ System initialized with {len(database.image_paths)} images using {model_info}", sample_images


def search_by_image(query_image, num_results: int):
    """Perform initial search with query image"""
    global current_results, selected_relevant, selected_non_relevant

    if query_image is None:
        return [None] * 20 + ["❌ Please upload a query image"] + ["—"] * 20

    if retrieval is None:
        return [None] * 20 + ["❌ Please initialize the system first"] + ["—"] * 20

    # Reset selections
    selected_relevant = set()
    selected_non_relevant = set()

    # Convert to PIL if needed
    if isinstance(query_image, np.ndarray):
        query_image = Image.fromarray(query_image)

    # Perform search
    current_results = retrieval.initial_search(query_image, top_k=num_results)

    # Prepare image outputs (up to 20 images)
    images = []
    for idx, (img_idx, score) in enumerate(current_results[:20]):
        img = database.get_image(img_idx)
        images.append(img)

    # Pad with None if less than 20 results
    while len(images) < 20:
        images.append(None)

    status = f"🔍 Found {len(current_results)} results (Iteration 0)"

    # Return images + status + radio defaults (all "—")
    return images + [status] + ["—"] * 20


def apply_feedback(*args):
    """Apply relevance feedback using Rocchio algorithm"""
    global current_results

    # args = 20 radio values + num_results
    radio_values = args[:20]
    num_results = int(args[20])

    if retrieval is None or len(current_results) == 0:
        return [None] * 20 + ["❌ Please perform an initial search first"] + ["—"] * 20

    # Parse radio selections
    relevant_indices = []
    non_relevant_indices = []

    for i, val in enumerate(radio_values):
        if i >= len(current_results):
            break
        if val == "✅ Relevant":
            img_idx = current_results[i][0]
            relevant_indices.append(img_idx)
        elif val == "❌ Non-Relevant":
            img_idx = current_results[i][0]
            non_relevant_indices.append(img_idx)

    if len(relevant_indices) == 0 and len(non_relevant_indices) == 0:
        # Return current images unchanged with warning
        images = []
        for idx, (img_idx, score) in enumerate(current_results[:20]):
            images.append(database.get_image(img_idx))
        while len(images) < 20:
            images.append(None)
        return images + ["⚠️ Please mark at least one image as Relevant or Non-Relevant"] + list(radio_values)

    # Apply Rocchio feedback
    current_results = retrieval.relevance_feedback(
        relevant_indices,
        non_relevant_indices,
        top_k=num_results
    )

    # Prepare image outputs
    images = []
    for idx, (img_idx, score) in enumerate(current_results[:20]):
        img = database.get_image(img_idx)
        images.append(img)

    while len(images) < 20:
        images.append(None)

    # Build status message
    if len(relevant_indices) == 0:
        status = f"🔄 Iteration {retrieval.iteration} | ⚠️ No relevant marked - try marking some as RELEVANT!"
    elif len(non_relevant_indices) == 0:
        status = f"🔄 Iteration {retrieval.iteration} | +{len(relevant_indices)} relevant"
    else:
        status = f"🔄 Iteration {retrieval.iteration} | +{len(relevant_indices)} relevant, -{len(non_relevant_indices)} non-relevant"

    # Return images + status + reset radios to "—"
    return images + [status] + ["—"] * 20


def update_rocchio_params(alpha, beta, gamma):
    """Update Rocchio algorithm parameters"""
    if retrieval is not None:
        retrieval.set_rocchio_params(alpha, beta, gamma)
        return f"✅ Parameters updated: α={alpha}, β={beta}, γ={gamma}"
    return "⚠️ Initialize system first"


def reset_search():
    """Reset the search state"""
    global current_results, selected_relevant, selected_non_relevant
    
    if retrieval is not None:
        retrieval.reset()
    
    current_results = []
    selected_relevant = set()
    selected_non_relevant = set()
    
    return None, "🔄 Search reset. Upload a new query image.", []


def search_by_database_index(index: int, num_results: int):
    """Search using an image from the database"""
    if database is None:
        return None, "❌ Please initialize the system first"
    if index < 0 or index >= len(database.image_paths):
        return None, f"❌ Invalid index. Valid range: 0 to {len(database.image_paths) - 1}"

    query_image = database.get_image(index)
    # Call search_by_image but only return gallery and status (not checkboxes)
    gallery, status, _ = search_by_image(query_image, num_results)
    return gallery, status


# Custom CSS for better appearance
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.title {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 14px;
    margin-bottom: 20px;
}

.feedback-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 15px;
    border-radius: 10px;
}

.result-gallery img {
    border-radius: 8px;
    transition: transform 0.2s;
}

.result-gallery img:hover {
    transform: scale(1.05);
}

footer {
    text-align: center;
    padding: 20px;
    color: #95a5a6;
}
"""


def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Anime Image Retrieval System") as demo:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">🎨 Anime Image Retrieval System</h1>
            <p style="color: #e0e0e0; margin: 5px 0 0 0;">CS419 - Assignment 02: Image Retrieval with Relevance Feedback</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: System Setup
            with gr.Tab("🔧 Setup"):
                gr.Markdown("### Initialize the System")
                gr.Markdown("Enter the path to your anime image dataset folder.")

                with gr.Row():
                    dataset_input = gr.Textbox(
                        label="Dataset Path",
                        placeholder="/path/to/anime/images",
                        value="./anime_images"
                    )
                    model_select = gr.Radio(
                        choices=[("CLIP (Recommended for Anime)", "clip"), ("ResNet50", "resnet50")],
                        value="clip",
                        label="Feature Model",
                        info="CLIP understands semantic content better (characters, scenes)"
                    )

                init_btn = gr.Button("🚀 Initialize System", variant="primary", size="lg")

                init_status = gr.Textbox(label="Status", interactive=False)
                sample_gallery = gr.Gallery(label="Sample Images from Database", columns=4, height=200)

                init_btn.click(
                    initialize_system,
                    inputs=[dataset_input, model_select],
                    outputs=[init_status, sample_gallery]
                )
                
                # Rocchio Parameters
                gr.Markdown("### ⚙️ Rocchio Algorithm Parameters")
                gr.Markdown("Adjust the weights for relevance feedback (Standard: α=1, β=0.75, γ=0.25)")
                
                with gr.Row():
                    alpha_slider = gr.Slider(0, 2, value=1.0, step=0.05, label="α (Original Query)")
                    beta_slider = gr.Slider(0, 2, value=0.75, step=0.05, label="β (Relevant)")
                    gamma_slider = gr.Slider(0, 2, value=0.25, step=0.05, label="γ (Non-Relevant)")
                
                params_btn = gr.Button("Update Parameters")
                params_status = gr.Textbox(label="Parameter Status", interactive=False)
                
                params_btn.click(
                    update_rocchio_params,
                    inputs=[alpha_slider, beta_slider, gamma_slider],
                    outputs=[params_status]
                )
            
            # Tab 2: Image Search
            with gr.Tab("🔍 Search"):
                # Query section
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Query Image")
                        query_image = gr.Image(label="Query Image", type="pil", height=250)
                        num_results = gr.Slider(5, 20, value=20, step=1, label="Number of Results")
                        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        gr.Markdown("### Instructions")
                        gr.Markdown("""
1. **Upload** a query image (e.g., an anime character)
2. **Click Search** to find similar images
3. **Mark each result** as Relevant or Non-Relevant
4. **Click Apply Feedback** to refine results
5. **Repeat** until satisfied!

💡 **Tip**: Mark at least 2-3 images as **Relevant** for best results!
""")
                        search_status = gr.Textbox(label="Status", interactive=False, value="Upload an image and click Search")

                gr.Markdown("---")
                gr.Markdown("### Search Results - Mark each image below:")

                # Create 20 image slots with radio buttons (4 columns x 5 rows)
                result_images = []
                result_radios = []

                for row in range(5):
                    with gr.Row():
                        for col in range(4):
                            idx = row * 4 + col
                            with gr.Column(min_width=150):
                                img = gr.Image(
                                    label=f"#{idx+1}",
                                    height=150,
                                    show_label=True,
                                    interactive=False
                                )
                                radio = gr.Radio(
                                    choices=["—", "✅ Relevant", "❌ Non-Relevant"],
                                    value="—",
                                    label="",
                                    interactive=True
                                )
                                result_images.append(img)
                                result_radios.append(radio)

                # Apply Feedback button
                gr.Markdown("---")
                feedback_btn = gr.Button("🚀 Apply Relevance Feedback", variant="primary", size="lg")

                # Wire up search - outputs: 20 images + status + 20 radios
                search_btn.click(
                    search_by_image,
                    inputs=[query_image, num_results],
                    outputs=result_images + [search_status] + result_radios
                )

                # Wire up feedback - inputs: 20 radios + num_results, outputs: 20 images + status + 20 radios
                feedback_btn.click(
                    apply_feedback,
                    inputs=result_radios + [num_results],
                    outputs=result_images + [search_status] + result_radios
                )
            
            # Tab 3: Browse Database
            with gr.Tab("📁 Browse Database"):
                gr.Markdown("### Browse and Search from Database")
                gr.Markdown("Enter an image index to use as query (useful for testing)")
                
                with gr.Row():
                    db_index = gr.Number(label="Image Index", value=0, precision=0)
                    db_num_results = gr.Slider(5, 50, value=20, step=1, label="Number of Results")
                    db_search_btn = gr.Button("Search by Index", variant="primary")
                
                db_status = gr.Textbox(label="Status", interactive=False)
                db_gallery = gr.Gallery(label="Results", columns=4, height=400)
                
                db_search_btn.click(
                    search_by_database_index,
                    inputs=[db_index, db_num_results],
                    outputs=[db_gallery, db_status]
                )
            
            # Tab 4: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This System
                
                This is an **Image Retrieval System with Relevance Feedback** developed for CS419 course.
                
                ### 🔧 Features Used
                
                **1. Feature Extraction:**
                - **CNN Features (ResNet50)**: Deep semantic features capturing image content
                - **Color Histogram (HSV)**: Color distribution features for anime's vibrant colors
                - Combined with weights: 70% CNN + 30% Color
                
                **2. Similarity Measure:**
                - **Cosine Similarity**: Measures angle between feature vectors
                
                **3. Relevance Feedback (Rocchio Algorithm):**
                
                ```
                q_new = α·q_old + β·(1/|Dr|)·Σ(d∈Dr) - γ·(1/|Dnr|)·Σ(d∈Dnr)
                ```
                
                Where:
                - `α` = Weight for original query (default: 1.0)
                - `β` = Weight for relevant documents (default: 0.75)
                - `γ` = Weight for non-relevant documents (default: 0.25)
                - `Dr` = Set of relevant documents
                - `Dnr` = Set of non-relevant documents
                
                ### 📚 References
                - CS419 Lecture Slides: Interactive Search with Relevance Feedback
                - Rocchio, J.J. (1971). Relevance feedback in information retrieval
                
                ### 🎓 Course Information
                - Course: CS419 - Introduction to Information Retrieval
                - University: VNU-HCM University of Science
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #eee;">
            <p style="color: #95a5a6; margin: 0;">
                CS419 Assignment 02 - Image Retrieval with Relevance Feedback<br>
                University of Science - VNU-HCM
            </p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=custom_css
    )
