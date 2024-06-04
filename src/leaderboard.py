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


CAPABILITY_COLUMNS = [
    "Grounding ⚡️",
    "Instruction Following 📝",
    "Planning 📅",
    "Reasoning 💡",
    "Refinement 🔩",
    "Safety ⚠️",
    "Theory of Mind 🤔",
    "Tool Usage 🛠️",
    "Multilingual 🇬🇫",
]

BGB_COLUMNS_MAPPING = {
    "model_name_or_path": "Model 🤗",
    "average": "Average",
    "grounding": "Grounding ⚡️",
    "instruction_following": "Instruction Following 📝",
    "planning": "Planning 📅",
    "reasoning": "Reasoning 💡",
    "refinement": "Refinement 🔩",
    "safety": "Safety ⚠️",
    "theory_of_mind": "Theory of Mind 🤔",
    "tool_usage": "Tool Usage 🛠️",
    "multilingual": "Multilingual 🇬🇫",
    "model_params": "Model Params (B)",
    "model_type": "Model Type",
}

# Use the values above as keys to create the values

BGB_COLUMN_TO_DATATYPE = {
    "Model 🤗": "markdown",
    "Model Params (B)": "number",
    "Model Type": "str",
    "Average": "number",
    "Grounding ⚡️": "number",
    "Instruction Following 📝": "number",
    "Planning 📅": "number",
    "Reasoning 💡": "number",
    "Refinement 🔩": "number",
    "Safety ⚠️": "number",
    "Theory of Mind 🤔": "number",
    "Tool Usage 🛠️": "number",
    "Multilingual 🇬🇫": "number",
}

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


def get_bgb_leaderboard_df(eval_df):
    df = eval_df.copy()
    # transform for leaderboard
    df["Model 🤗"] = df["Model 🤗"].apply(process_model)
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


def create_bgb_leaderboard_table(eval_df):
    # get dataframe
    bgb_leaderboard_df = get_bgb_leaderboard_df(eval_df)
    
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
            label="Capabilities 📊",
            value=CAPABILITY_COLUMNS,
            choices=CAPABILITY_COLUMNS,
            info="☑️ Select the capabilities to display",
            elem_id="columns-checkboxes",
        )
    
    # create table
    bgb_leaderboard_table = gr.components.Dataframe(
        value=bgb_leaderboard_df[list(BGB_COLUMNS_MAPPING.values())],
        datatype=list(BGB_COLUMN_TO_DATATYPE.values()),
        headers=list(BGB_COLUMNS_MAPPING.keys()),
        elem_id="leaderboard-table",
    )

    return search_bar, columns_checkboxes, bgb_leaderboard_table
