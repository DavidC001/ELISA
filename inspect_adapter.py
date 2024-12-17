import os
import glob
import torch
import gradio as gr
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import warnings

from configuration import load_yaml_config
from inference import InferencePipeline, InferenceSample
from llava_finetune.utils import (
    draw_shapes,
)  # Ensure this points to the updated function

# Disable warnings
warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # non-interactive backend
# Disable interactive matplotlib
plt.ioff()

# Load configuration
config = load_yaml_config("config.yaml")

# Retrieve available models from the 'models/' directory
model_paths = glob.glob("models/*.pth")
model_names = [os.path.basename(p).replace(".pth", "") for p in model_paths]

# Initialize a dictionary to cache loaded models
pipelines: dict[str, InferencePipeline] = {}
global_data = {
    "model": None,
    "token_similarities": None,
}


def load_selected_model(model_name):
    """
    Load and cache the selected model.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = InferencePipeline(config, model_name)
    return model


def process_image(
    model_name, image, softmax=True, temperature=2.0, cosine_similarity=False
):
    """
    Process the uploaded image to generate segmentation masks and embeddings.

    Args:
        model_name (str): Selected model name.
        image (PIL.Image.Image): Uploaded image.
        softmax (bool): Flag to use softmax for token similarity calculation.
        temperature (float): Temperature value for token similarity calculation.
        cosine_similarity (bool): Flag to use cosine similarity.

    Returns:
        Tuple: Processed image, status message and mask choices.
    """
    if model_name not in pipelines:
        try:
            pipelines[model_name] = load_selected_model(model_name)
        except Exception as e:
            return None, f"Error loading model '{model_name}': {str(e)}", None, None

    model = pipelines[model_name]

    if image is None:
        return None, "Please upload an image.", None, None

    # Save the uploaded image temporarily
    tmp_path = "inference/temp_in.png"
    image.save(tmp_path)

    # Preprocess the image to obtain segmentation embeddings and shapes
    try:
        results = model.token_similarity(
            [InferenceSample(query="", image=tmp_path)],
            softmax=softmax,
            temperature=temperature,
            cosine_similarity=cosine_similarity,
        )
        result = next(results)
    except Exception as e:
        return None, f"Error during preprocessing: {str(e)}", None, None

    masks = result["masks"]
    token_similarities = result["token_similarities"]
    embs_similarities = result["embs_similarities"]

    image = draw_shapes(Image.open(tmp_path), masks, enumerate_masks=True)

    # global_data["masks"] = masks
    global_data["model"] = model_name
    global_data["token_similarities"] = token_similarities
    global_data["embs_similarities"] = embs_similarities

    mask_choices = [f"Mask {i+1}" for i in range(len(masks))]

    # Generate the image of the intra-mask similarity
    fig, ax = plt.subplots(figsize=(16, 12))
    cax = ax.matshow(embs_similarities.cpu().numpy(), cmap="viridis")
    fig.colorbar(cax)
    ax.set_title("Intra-Mask Similarity Matrix")
    plt.tight_layout()
    plot_path = "output/intra_mask_similarity.png"
    plt.savefig(plot_path)
    plt.close(fig)

    return (
        image,
        "Image processed successfully.",
        gr.update(choices=mask_choices),
        Image.open(plot_path),
    )


def get_top_tokens(selected_mask, num_tokens=10):
    """
    Retrieve and display the top similar tokens for the selected mask, visualized as a bar chart.
    """
    if global_data["token_similarities"] is None:
        return "Please process an image first."

    if selected_mask is None:
        return "Please select a mask first."

    mask_idx = int(selected_mask.split()[-1]) - 1
    token_similarities = global_data["token_similarities"]

    # Retrieve the top similar tokens based on the selected mask
    top_values, top_indices = torch.topk(token_similarities[mask_idx], num_tokens)
    # Decode token indices
    tokens = [
        pipelines[global_data["model"]].tokenizer.decode([int(i)]) for i in top_indices
    ]

    # Convert tensors to CPU and numpy for plotting if needed
    top_values = top_values.cpu().numpy() if top_values.is_cuda else top_values.numpy()

    # Create a horizontal bar chart visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    y_pos = range(len(tokens))

    # Plot horizontal bars
    ax.barh(y_pos, top_values[::-1], align="center", color="skyblue")

    # Set tick labels and invert y-axis so top token is at the top
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens[::-1])
    ax.invert_yaxis()  # labels read top-to-bottom

    # Labeling
    ax.set_xlabel("Similarity Score")
    ax.set_title(f"Top {num_tokens} Similar Tokens for Mask {mask_idx+1}")

    fig.tight_layout()

    # Save the plot as an image
    plot_path = "output/top_tokens.png"
    plt.savefig(plot_path)
    plt.close(fig)

    return Image.open(plot_path)


if __name__ == "__main__":
    # Define the Gradio interface
    with gr.Blocks(
        title="Segmentation Mask Embeddings Explorer with Model Selection"
    ) as demo:
        gr.Markdown("# 🖼️ Segmentation Mask Embeddings Explorer")

        with gr.Row():
            with gr.Column():
                # Model selection dropdown
                model_dropdown = gr.Dropdown(
                    choices=model_names,
                    label="🔍 Select Model",
                    value=model_names[0] if model_names else None,
                    info="Choose a model from the available list.",
                )
                # Image upload component
                image_input = gr.Image(type="pil", label="🖼️ Upload Image")
                # checkbox to use softmax
                softmax_checkbox = gr.Checkbox(label="Use Softmax", value=True)
                # Temperature slider for token similarity
                temperature_slider = gr.Slider(
                    label="🌡️ Temperature",
                    minimum=0.1,
                    maximum=5.0,
                    value=2.0,
                    step=0.1,
                    info="Smoothing factor for token similarity calculation.",
                )
                # checkbox to use cosine similarity
                cosine_similarity_checkbox = gr.Checkbox(
                    label="Use Cosine Similarity", value=False
                )
                # Process image button
                process_button = gr.Button("📄 Process Image")

                # Status message textbox
                status_message = gr.Textbox(label="🛈 Status", interactive=False)

            with gr.Column():
                # Display processed image with masks
                processed_image_output = gr.Image(
                    label="🖼️ Image with Masks", type="pil"
                )
                # Mask selection dropdown
                mask_dropdown = gr.Dropdown(label="🎯 Select Mask", choices=[])
                # Top tokens button selection
                num_tokens_input = gr.Number(
                    label="🔢 Number of Tokens", value=10, minimum=1
                )
                # Button to get top tokens
                get_tokens_button = gr.Button("🔍 Get Top Similar Tokens")

        with gr.Row():
            with gr.Column():
                # Output textbox for tokens
                token_output = gr.Image(label="📊 Top Tokens Chart", type="pil")
            with gr.Column():
                # Output textbox for tokens intra-mask similarity
                intra_mask_output = gr.Image(label="📊 Intra-Mask Similarity Chart", type="pil")

        gr.Markdown(
            """
            Welcome! This tool allows you to:
            1. **Select a model** from the available options.
            2. **Upload an image** to process.
            3. **View segmentation masks** overlaid on the image.
            4. **Select a mask** to explore the top most similar tokens in the model's vocabulary based on the mask's embedding.

            **Instructions:**
            - Choose a model from the dropdown menu.
            - Upload an image containing identifiable objects.
            - Click "Process Image" to visualize segmentation masks.
            - Select a mask from the dropdown to view related tokens.
            """
        )

        # Define the button interactions
        process_button.click(
            fn=process_image,
            inputs=[
                model_dropdown,
                image_input,
                softmax_checkbox,
                temperature_slider,
                cosine_similarity_checkbox,
            ],
            outputs=[
                processed_image_output,
                status_message,
                mask_dropdown,
                intra_mask_output,
            ],
        )

        get_tokens_button.click(
            fn=get_top_tokens,
            inputs=[mask_dropdown, num_tokens_input],
            outputs=[token_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
