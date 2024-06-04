import gradio as gr

from src.assets import custom_css

from src.content import ABOUT, CITATION_BUTTON, CITATION_BUTTON_LABEL, LOGO, TITLE
from src.leaderboard import create_leaderboard_table, create_bgb_leaderboard_table
from src.llm_perf import get_llm_perf_df, get_eval_df
from src.panel import (
    create_select_callback,
)


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


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(LOGO, elem_classes="logo")
    gr.HTML(TITLE, elem_classes="title")

    with gr.Tabs(elem_classes="tabs"):
        hardware = "Leaderboard 🏅"
        machine = "1xA10"
        machine_textbox = gr.Textbox(value=machine, visible=False)

        if BGB:
            eval_df = get_eval_df("gpt-4-turbo-2024-04-09")
        else:
            eval_df = get_llm_perf_df(machine=machine)
        # Leaderboard
        with gr.TabItem("Leaderboard 🏅", id=0):
            if BGB:
                search_bar, columns_checkboxes, leaderboard_table = create_bgb_leaderboard_table(eval_df)
            else:
                search_bar, columns_checkboxes, leaderboard_table = create_leaderboard_table(eval_df)

        create_select_callback(
            # inputs
            machine_textbox,
            # interactive
            columns_checkboxes,
            search_bar,
            # outputs
            leaderboard_table,
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
