import gradio as gr

from src.utils import model_hyperlink, process_score

LEADERBOARD_COLUMN_TO_DATATYPE = {
    # open llm
    "Model 🤗": "markdown",
    "Experiment 🧪": "str",
    # primary measurements
    "Prefill (s)": "number",
    "Decode (tokens/s)": "number",
    "Memory (MB)": "number",
    "Energy (tokens/kWh)": "number",
    # deployment settings
    "Backend 🏭": "str",
    "Precision 📥": "str",
    "Quantization 🗜️": "str",
    "Attention 👁️": "str",
    "Kernel ⚛️": "str",
    # additional measurements
    # "Reserved Memory (MB)": "number",
    # "Used Memory (MB)": "number",
    "Open LLM Score (%)": "number",
    "End-to-End (s)": "number",
    "Architecture 🏛️": "str",
    "Params (B)": "number",
}

PRIMARY_COLUMNS = [
    "Model 🤗",
    "Experiment 🧪",
    "Prefill (s)",
    "Decode (tokens/s)",
    "Memory (MB)",
    "Energy (tokens/kWh)",
    "Open LLM Score (%)",
]


def process_model(model_name):
    link = f"https://huggingface.co/{model_name}"
    return model_hyperlink(link, model_name)


def get_leaderboard_df(llm_perf_df):
    df = llm_perf_df.copy()
    # transform for leaderboard
    df["Model 🤗"] = df["Model 🤗"].apply(process_model)
    # process quantization for leaderboard
    df["Open LLM Score (%)"] = df.apply(lambda x: process_score(x["Open LLM Score (%)"], x["Quantization 🗜️"]), axis=1)
    return df


def create_leaderboard_table(llm_perf_df):
    # get dataframe
    leaderboard_df = get_leaderboard_df(llm_perf_df)

    # create search bar
    with gr.Row():
        search_bar = gr.Textbox(
            label="Model 🤗",
            info="🔍 Search for a model name",
            elem_id="search-bar",
        )
    # create checkboxes
    with gr.Row():
        columns_checkboxes = gr.CheckboxGroup(
            label="Columns 📊",
            value=PRIMARY_COLUMNS,
            choices=list(LEADERBOARD_COLUMN_TO_DATATYPE.keys()),
            info="☑️ Select the columns to display",
            elem_id="columns-checkboxes",
        )
    # create table
    leaderboard_table = gr.components.Dataframe(
        value=leaderboard_df[PRIMARY_COLUMNS],
        datatype=list(LEADERBOARD_COLUMN_TO_DATATYPE.values()),
        headers=list(LEADERBOARD_COLUMN_TO_DATATYPE.keys()),
        elem_id="leaderboard-table",
    )

    return search_bar, columns_checkboxes, leaderboard_table
