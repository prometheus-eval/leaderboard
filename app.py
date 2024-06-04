import gradio as gr
from gradio_leaderboard import ColumnFilter, Leaderboard, SelectColumns

from src.assets import custom_css
from src.content import ABOUT, BGB_LOGO, BGB_TITLE, CITATION_BUTTON, CITATION_BUTTON_LABEL, LOGO, TITLE
from src.leaderboard import (
    BGB_COLUMN_MAPPING,
    BGB_COLUMN_TO_DATATYPE,
    CAPABILITY_COLUMNS,
    create_bgb_leaderboard_table,
    create_leaderboard_table,
    get_bgb_leaderboard_df,
)
from src.llm_perf import get_eval_df, get_llm_perf_df
from src.panel import create_select_callback

BGB = True

# prometheus-eval/prometheus-bgb-8x7b-v2.0

# def init_leaderboard():
#     machine = "1xA10"
#     open_llm_perf_df = get_llm_perf_df(machine=machine)
#     search_bar, columns_checkboxes, leaderboard_table = create_leaderboard_table(open_llm_perf_df)
#     return machine, search_bar, columns_checkboxes, leaderboard_table


EVAL_MODELS = [
    "gpt-4-turbo-2024-04-09",
    "prometheus-bgb-8x7b-v2.0",
]

EVAL_MODEL_TABS = {
    "gpt-4-turbo-2024-04-09": "GPT-4 as a Judge 🏅",
    "prometheus-bgb-8x7b-v2.0": "Prometheus as a Judge 🏅",
}


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(BGB_LOGO, elem_classes="logo")
    gr.HTML(BGB_TITLE, elem_classes="title")
    # gr.HTML(BGB_LOGO_AND_TITLE, elem_classes="title")

    with gr.Tabs(elem_classes="tabs"):

        for idx, eval_model in enumerate(EVAL_MODELS):
            tab_name = EVAL_MODEL_TABS[eval_model]

            # Previous code without gradio_leaderboard

            # machine = eval_model
            # machine_textbox = gr.Textbox(value=eval_model, visible=False)

            # if BGB:
            #     eval_df = get_eval_df(eval_model_name=eval_model)
            # else:
            #     eval_df = get_llm_perf_df(machine=machine)
            # # Leaderboard
            # with gr.TabItem(tab_name, id=idx):
            #     if BGB:
            #         search_bar, columns_checkboxes, type_checkboxes, param_slider, leaderboard_table = create_bgb_leaderboard_table(eval_df)
            #     else:
            #         search_bar, columns_checkboxes, type_checkboxes, param_slider, leaderboard_table = (
            #             create_leaderboard_table(eval_df)
            #         )

            # create_select_callback(
            #     # inputs
            #     machine_textbox,
            #     # interactive
            #     columns_checkboxes,
            #     search_bar,
            #     type_checkboxes,
            #     param_slider,
            #     # outputs
            #     leaderboard_table,
            # )
            with gr.TabItem(tab_name, id=idx):

                eval_df = get_eval_df(eval_model_name=eval_model)
                eval_df = get_bgb_leaderboard_df(eval_df)

                ordered_columns = [
                    "Model 🤗",
                    "Average",
                    "Grounding ⚡️",
                    "Instruction Following 📝",
                    "Planning 📅",
                    "Reasoning 💡",
                    "Refinement 🔩",
                    "Safety ⚠️",
                    "Theory of Mind 🤔",
                    "Tool Usage 🛠️",
                    "Multilingual 🇬🇫",
                    "Model Type",
                    "Model Params (B)",
                ]

                ordered_columns_types = [
                    "markdown",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "number",
                    "text",
                    "number",
                ]

                eval_df = eval_df[ordered_columns]

                Leaderboard(
                    value=eval_df,
                    datatype=ordered_columns_types,
                    select_columns=SelectColumns(
                        default_selection=ordered_columns,
                        cant_deselect=["Model 🤗", "Model Type", "Model Params (B)"],
                        label="Select Columns to Display:",
                    ),
                    search_columns=["Model 🤗"],
                    # hide_columns=["model_name_for_query", "Model Size"],
                    filter_columns=[
                        ColumnFilter("Model Type", type="checkboxgroup", label="Model types"),
                        ColumnFilter(
                            "Model Params (B)",
                            min=0,
                            max=150,
                            default=[0, 150],
                            type="slider",
                            label="Model Params (B)",
                        ),
                    ],
                )

        ####################### ABOUT TAB #######################
        with gr.TabItem("About 📖", id=3):
            gr.Markdown(ABOUT, elem_classes="descriptive-text")

    ####################### CITATION
    with gr.Row():
        with gr.Accordion("📙 Citation", open=False):
            citation_button = gr.Textbox(
                value=CITATION_BUTTON,
                label=CITATION_BUTTON_LABEL,
                elem_id="citation-button",
                show_copy_button=True,
            )

if __name__ == "__main__":
    # Launch demo
    demo.queue().launch()
