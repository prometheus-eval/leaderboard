import os

import pandas as pd
from pathlib import Path
from src.utils import process_kernels, process_quantizations
from src.model_list import get_all_model_list, MODEL_SHORT_TO_LONG, MODEL_MAPPING


COLUMNS_MAPPING = {
    "config.name": "Experiment 🧪",
    "config.backend.model": "Model 🤗",
    # primary measurements
    "report.prefill.latency.p50": "Prefill (s)",
    "report.per_token.latency.p50": "Per Token (s)",
    "report.decode.throughput.value": "Decode (tokens/s)",
    "report.decode.efficiency.value": "Energy (tokens/kWh)",
    "report.decode.memory.max_allocated": "Memory (MB)",
    # deployment settings
    "config.backend.name": "Backend 🏭",
    "config.backend.torch_dtype": "Precision 📥",
    "quantization": "Quantization 🗜️",
    "attention": "Attention 👁️",
    "kernel": "Kernel ⚛️",
    # additional information
    "architecture": "Architecture 🏛️",
    "prefill+decode": "End-to-End (s)",
    "Average ⬆️": "Open LLM Score (%)",
    "#Params (B)": "Params (B)",
}
SORTING_COLUMNS = ["Open LLM Score (%)", "Decode (tokens/s)", "Prefill (s)"]
SUBSETS = ["unquantized", "awq", "bnb", "gptq"]
SORTING_ASCENDING = [False, True, False]

BGB_SORTING_COLUMNS = ["Average"]

# Use the above capabilities to create the columns
BGB_COLUMNS_MAPPING = {
    "model_name_or_path": "Model 🤗",
    "model_params": "Model Params (B)",
    "model_type": "Model Type",
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
}


def get_raw_llm_perf_df(machine: str = "1xA10"):
    dfs = []
    for subset in SUBSETS:
        try:
            dfs.append(
                pd.read_csv(f"hf://datasets/optimum-benchmark/llm-perf-leaderboard/perf-df-{subset}-{machine}.csv")
            )
        except Exception:
            print(f"Subset {subset} for machine {machine} not found")

    perf_df = pd.concat(dfs)
    llm_df = pd.read_csv("hf://datasets/optimum-benchmark/llm-perf-leaderboard/llm-df.csv")

    llm_perf_df = pd.merge(llm_df, perf_df, left_on="Model", right_on="config.backend.model")

    return llm_perf_df


def processed_llm_perf_df(llm_perf_df):
    # some assertions
    assert llm_perf_df["config.scenario.input_shapes.batch_size"].nunique() == 1
    assert llm_perf_df["config.scenario.input_shapes.sequence_length"].nunique() == 1
    assert llm_perf_df["config.scenario.generate_kwargs.max_new_tokens"].nunique() == 1
    assert llm_perf_df["config.scenario.generate_kwargs.min_new_tokens"].nunique() == 1
    # fix couple stuff
    llm_perf_df.dropna(subset=["report.decode.latency.p50"], inplace=True)
    llm_perf_df["config.name"] = llm_perf_df["config.name"].str.replace("flash_attention_2", "fa2")
    llm_perf_df["prefill+decode"] = llm_perf_df["report.prefill.latency.p50"] + (
        llm_perf_df["report.decode.latency.p50"]
    )
    # llm_perf_df["architecture"] = llm_perf_df["config.backend.model"].apply(
    #     process_architectures
    # )
    llm_perf_df["architecture"] = llm_perf_df["Architecture"]
    llm_perf_df["attention"] = (
        llm_perf_df["config.backend.attn_implementation"]
        .str.replace("flash_attention_2", "FAv2")
        .str.replace("eager", "Eager")
        .str.replace("sdpa", "SDPA")
    )
    llm_perf_df["quantization"] = llm_perf_df.apply(process_quantizations, axis=1)
    llm_perf_df["kernel"] = llm_perf_df.apply(process_kernels, axis=1)
    # round numerical columns
    llm_perf_df = llm_perf_df.round(
        {
            "report.prefill.latency.p50": 3,
            "report.decode.latency.p50": 3,
            "report.decode.throughput.value": 3,
            "report.decode.efficiency.value": 3,
            "report.decode.memory.max_allocated": 3,
            "Average ⬆️": 3,
            "prefill+decode": 3,
            "#Params (B)": 3,
        }
    )
    # filter columns
    llm_perf_df = llm_perf_df[list(COLUMNS_MAPPING.keys())]
    # rename columns
    llm_perf_df.rename(columns=COLUMNS_MAPPING, inplace=True)
    # sort by metric
    llm_perf_df.sort_values(
        by=SORTING_COLUMNS,
        ascending=SORTING_ASCENDING,
        inplace=True,
    )

    return llm_perf_df


def get_llm_perf_df(machine: str = "1xA10"):
    if os.path.exists(f"llm-perf-leaderboard-{machine}.csv"):
        llm_perf_df = pd.read_csv(f"llm-perf-leaderboard-{machine}.csv")
    else:
        llm_perf_df = get_raw_llm_perf_df(machine)
        llm_perf_df = processed_llm_perf_df(llm_perf_df)
        llm_perf_df.to_csv(f"llm-perf-leaderboard-{machine}.csv", index=False)

    return llm_perf_df


def get_eval_df(eval_model_name: str):

    assert eval_model_name in ["gpt-4-turbo-2024-04-09", "prometheus-bgb-8x7b-v2.0"]

    base_dir = Path(__file__).parent.parent / "data"
    filepath = base_dir / f"bgb-leaderboard-{eval_model_name}.pkl"
    # For debugging
    csv_filepath = base_dir / f"bgb-leaderboard-{eval_model_name}.csv"
    
    def change_model_name(model_name: str):
        # TODO: Hard code models with different names        
        model_name_or_path = MODEL_SHORT_TO_LONG.get(model_name, model_name)
        if model_name == "qwen/qwen-110b-chat":
            model_name_or_path = "Qwen/Qwen1.5-110B-Chat"
            
        if model_name_or_path.endswith("-hjpark"):
            model_name_or_path = model_name_or_path.replace("-hjpark", "")
        
        return model_name_or_path

    if os.path.exists(filepath) and False:
        eval_df = pd.read_pickle(filepath)
    else:
        # Process the df
        raw_filepath = base_dir / f"eval_by_{eval_model_name}.csv"
        eval_df = pd.read_csv(raw_filepath)

        eval_df['model_name_or_path'] = eval_df['model_name'].apply(lambda x: change_model_name(x))
        eval_df.drop(columns=['model_name'], inplace=True)

        eval_df['model_params'] = eval_df['model_name_or_path'].apply(lambda x: MODEL_MAPPING.get(x, ['Unknown', 'Unknown'])[0])
        eval_df['model_type'] = eval_df['model_name_or_path'].apply(lambda x: MODEL_MAPPING.get(x, ['Unknown', 'Unknown'])[1])

        capabilities = [
            "grounding",
            "instruction_following",
            "planning",
            "reasoning",
            "refinement",
            "safety",
            "theory_of_mind",
            "tool_usage",
            "multilingual",
        ]
        
        # Make the average of the capabilities
        eval_df['average'] = eval_df[capabilities].mean(axis=1)

        # Round to 3 decimal places for capabilities and average
        eval_df = eval_df.round(
            {
                "average": 3,
                "grounding": 3,
                "instruction_following": 3,
                "planning": 3,
                "reasoning": 3,
                "refinement": 3,
                "safety": 3,
                "theory_of_mind": 3,
                "tool_usage": 3,
                "multilingual": 3,
            }
        )

        # print(eval_df[eval_df['model_params'] == 'Unknown'])
        eval_df.rename(columns=BGB_COLUMNS_MAPPING, inplace=True)
        
        eval_df.sort_values(
            by=BGB_SORTING_COLUMNS,
            ascending=False,
            inplace=True,
        )

        eval_df.to_pickle(str(filepath))
        eval_df.to_csv(str(csv_filepath), index=False)
    # import pdb; pdb.set_trace()

    return eval_df

if __name__ == "__main__":
    get_eval_df("gpt-4-turbo-2024-04-09")
    get_eval_df("prometheus-bgb-8x7b-v2.0")
