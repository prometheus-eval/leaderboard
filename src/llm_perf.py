import os

import pandas as pd

from .utils import process_kernels, process_quantizations

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
