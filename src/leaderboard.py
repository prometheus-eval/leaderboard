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


BGB_COLUMN_MAPPING = {
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


BGB_COLUMN_TO_DATATYPE = {
    "Model 🤗": "markdown",
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
    "Model Params (B)": "number",
    "Model Type": "str",
}


def process_model(model_name):
    link = f"https://huggingface.co/{model_name}"
    return model_hyperlink(link, model_name)


# TODO: Process base, chat, proprietary models differently
def process_bgb_model(row):
    model_name = row.iloc[0]
    model_type = row.iloc[1]

    if model_type == "Base" or model_type == "Chat":
        link = f"https://huggingface.co/{model_name}"
        return model_hyperlink(link, model_name)
    elif model_type == "Proprietary":

        api_model_2_link = {
            "gpt-3.5-turbo-1106": "https://platform.openai.com/docs/models/gpt-3-5",
            "gpt-3.5-turbo-0125": "https://platform.openai.com/docs/models/gpt-3-5",
            "gpt-4-0125-preview": "https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
            "gpt-4-1106-preview": "https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
            "gpt-4-turbo-2024-04-09": "https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4",
            "gpt-4o-2024-05-13": "https://openai.com/index/hello-gpt-4o/",
            "claude-3-haiku-20240307": "https://www.anthropic.com/news/claude-3-family",
            "claude-3-opus-20240229": "https://www.anthropic.com/news/claude-3-family",
            "claude-3-sonnet-20240229": "https://www.anthropic.com/news/claude-3-family",
            "mistral-large": "https://mistral.ai/news/mistral-large/",
            "mistral-medium": "https://mistral.ai/news/la-plateforme/",
            "gemini-1.0-pro": "https://deepmind.google/technologies/gemini/pro/",
            "gemini-pro-1.5": "https://deepmind.google/technologies/gemini/pro/",
            "google/gemini-flash-1.5": "https://deepmind.google/technologies/gemini/flash/",
        }

        link = api_model_2_link[model_name]
        return model_hyperlink(link, model_name)

    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")


def get_leaderboard_df(llm_perf_df):
    df = llm_perf_df.copy()
    # transform for leaderboard
    df["Model 🤗"] = df["Model 🤗"].apply(process_bgb_model)
    # process quantization for leaderboard
    df["Open LLM Score (%)"] = df.apply(lambda x: process_score(x["Open LLM Score (%)"], x["Quantization 🗜️"]), axis=1)
    return df


def get_bgb_leaderboard_df(eval_df):
    df = eval_df.copy()
    # transform for leaderboard
    df["Model 🤗"] = df[["Model 🤗", "Model Type"]].apply(process_bgb_model, axis=1)
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

    with gr.Row():
        type_checkboxes = gr.CheckboxGroup(
            label="Model Type",
            value=["Base", "Chat", "Proprietary"],
            choices=["Base", "Chat", "Proprietary"],
            info="☑️ Select the capabilities to display",
            elem_id="type-checkboxes",
        )

    with gr.Row():
        param_slider = gr.Slider(
            minimum=0, maximum=150, value=7, step=1, interactive=True, label="Model Params (B)", elem_id="param-slider"
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
        value=bgb_leaderboard_df[list(BGB_COLUMN_MAPPING.values())],
        datatype=list(BGB_COLUMN_TO_DATATYPE.values()),
        headers=list(BGB_COLUMN_MAPPING.keys()),
        elem_id="leaderboard-table",
    )

    return search_bar, columns_checkboxes, type_checkboxes, param_slider, bgb_leaderboard_table
