import os
import glob
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt

from inference import InferencePipeline, InferenceSample, load_yaml_config
from llava_finetune.utils import draw_shapes

plt.ioff()  # Disable interactive matplotlib

# Load configuration
config = load_yaml_config("config.yaml")

# Get the list of available models
model_paths = glob.glob("models/*.pth")
model_names = [os.path.basename(p).replace(".pth", "") for p in model_paths]

def load_pipeline(model_name):
    pipeline = InferencePipeline(config, model_name)
    return pipeline

# Cache pipelines
pipelines = {}

def inference_fn(model_name, query, image, max_new_tokens, n_beams, temperature, repeat_penalty):
    if image is None or query.strip() == "":
        return "Please provide both an image and a query.", None

    if model_name not in pipelines:
        pipelines[model_name] = load_pipeline(model_name)

    pipeline : InferencePipeline = pipelines[model_name]

    # Temporary save for inference
    tmp_image_path = "inference/temp_upload.png"
    image.save(tmp_image_path)

    data = [InferenceSample(query=query, image=tmp_image_path)]

    results = pipeline.inference(data, max_new_tokens=max_new_tokens, n_beams=n_beams, temperature=temperature, repeat_penalty=repeat_penalty)
    result = next(results)

    orig_image = Image.open(tmp_image_path).convert("RGBA")
    processed_image = draw_shapes(orig_image, result["masks"], mask_names=[f"<SEG_MASK_{idx+1}>" for idx in result["chosen_tokens"]])

    return result["gen_text"], processed_image

with gr.Blocks(title="LLaVA Model Inference Interface") as demo:
    gr.Markdown("# LLaVA Visual & Language Reasoning")
    gr.Markdown(
        "Welcome to the LLaVA inference interface! This tool lets you upload an image and ask a question about it. "
        "The model will attempt to answer based on the visual content and your query."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input")

            model_dropdown = gr.Dropdown(
                choices=model_names,
                value=model_names[0] if model_names else None,
                label="Select Model",
                info="Choose from the available finetuned models."
            )
            image_input = gr.Image(
                type="pil", 
                label="Upload Image", 
            )
            query_input = gr.Textbox(
                label="Your Query", 
                placeholder="e.g. 'What objects are in the image?'"
            )

            with gr.Accordion("Advanced Options", open=False):
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=100,
                    step=1,
                    label="Max Tokens",
                    info="Maximum number of tokens to generate."
                )
                n_beams_slider = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    value=5, 
                    step=1, 
                    label="Number of Beams (Search Width)",
                    info="Higher values may improve result quality but take longer."
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=2.0, 
                    value=0.8, 
                    step=0.1, 
                    label="Temperature",
                    info="Higher values make the model more creative."
                )
                repeat_penalty_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=2.0, 
                    value=2.0, 
                    step=0.1, 
                    label="Repetition Penalty",
                    info="Higher values reduce repeated tokens in the output."
                )
            
            with gr.Row():
                run_button = gr.Button("Run Inference", variant="primary")
                clear_button = gr.Button("Clear")

            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    [model_names[0] if model_names else None, "What vehicle should I sleep in?", "inference/2593366765_589ca5148e_o.jpg"],
                    [model_names[0] if model_names else None, "Where is the van?", "inference/2593366765_589ca5148e_o.jpg"],
                    [model_names[0] if model_names else None, "Is there a ladder in this image?", "inference/2593366765_589ca5148e_o.jpg"],
                ],
                inputs=[model_dropdown, query_input, image_input]
            )

        with gr.Column(scale=1):
            gr.Markdown("## Output")
            answer_output = gr.Textbox(
                label="Answer", 
                interactive=False, 
                placeholder="The answer from the model will appear here."
            )
            processed_image_output = gr.Image(
                label="Processed Image", 
                type="pil", 
                visible=True
            )

    # Button actions
    run_button.click(
        fn=inference_fn,
        inputs=[model_dropdown, query_input, image_input, max_new_tokens, n_beams_slider, temperature_slider, repeat_penalty_slider],
        outputs=[answer_output, processed_image_output]
    )

    def clear_all():
        return None, None, "", 5, None, None

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[model_dropdown, image_input, query_input, n_beams_slider, answer_output, processed_image_output],
        queue=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
