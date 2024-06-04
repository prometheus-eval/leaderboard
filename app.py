import gradio as gr

from src.assets import custom_css
from src.content import ABOUT, CITATION_BUTTON, CITATION_BUTTON_LABEL, LOGO, TITLE
from src.leaderboard import create_leaderboard_table
from src.llm_perf import get_llm_perf_df
from src.panel import create_select_callback

demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(LOGO, elem_classes="logo")
    gr.HTML(TITLE, elem_classes="title")

    ####################### HARDWARE TABS #######################
    with gr.Tabs(elem_classes="tabs"):
        hardware = "Leaderboard 🏅"
        machine = "1xA10"
        machine_textbox = gr.Textbox(value=machine, visible=False)

        open_llm_perf_df = get_llm_perf_df(machine=machine)
        with gr.TabItem("Leaderboard 🏅", id=0):
            search_bar, columns_checkboxes, leaderboard_table = create_leaderboard_table(open_llm_perf_df)

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
