import gradio as gr
from modules.dim_reduction import get_cols, calculate_dim_reduction

# ======================================================================================
#                               GRADIO INTERFACE
# ======================================================================================

with gr.Blocks() as demo:
    gr.Markdown("## Feature Selection Companion")
    
    with gr.Row():

        # for file upload
        with gr.Column():
            file_in = gr.File(label="Upload CSV File")
        
        # for thresholds selection
        with gr.Column():
            variance_threshold = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Variance Threshold")
            mvr_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Missing Value Ratio Threshold")
            corr_threshold = gr.Slider(0.0, 1.0, value=0.75, step=0.01, label="Correlation Threshold")

        # for target column selection and buttons        
        with gr.Column():
            get_features_button = gr.Button("Get Features")
            target_col = gr.Dropdown(visible=False)
            analyze_button = gr.Button(visible=False)

    gr.Markdown("## Analysis Results")

    # for consolidated features table
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🚩Consolidated Feature Flags")
            consolidated_table = gr.DataFrame(headers=["Feature", "Flag Count"], datatype=["str", "number"])

    with gr.Row():
        # for variance thresholding results
        with gr.Column():
            gr.Markdown("### 📈Variance Thresholding")
            variance_fig = gr.Plot()
            var_summary_table = gr.DataFrame(headers=["Feature", "Variance", "Flagged"], datatype=["str", "number", "str"])
            
        with gr.Column():
            # for missing value ratio results
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📈Missing Value Ratio")
                    mvr_fig = gr.Plot()
                    mvr_summary_table = gr.DataFrame(headers=["Feature", "Missing Value Ratio", "Flagged"], datatype=["str", "number", "str"])

            # for correlation results
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊Correlation Analysis")
                    corr_fig = gr.Plot()
                    corr_summary_table = gr.DataFrame()

# ======================================================================================
#                               BUTTON CALLBACKS
# ======================================================================================

    get_features_button.click(
        fn=get_cols,
        inputs=[file_in],
        outputs=[target_col, analyze_button]
    )

    analyze_button.click(
        fn=calculate_dim_reduction,
        inputs=[file_in, variance_threshold, mvr_threshold, corr_threshold, target_col],
        outputs=[consolidated_table, var_summary_table, variance_fig, mvr_summary_table, mvr_fig, corr_summary_table, corr_fig]
    )    

demo.launch()