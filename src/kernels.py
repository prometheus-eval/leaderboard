import gradio as gr
import pandas as pd
import plotly.express as px

QUANT_DATA = [
    # open llm
    "Model 🤗",
    "DType 📥",
    "Backend 🏭",
    "Params (B)",
    "Architecture 🏛️",
    "Open LLM Score (%)",
    # deployment settings
    "DType 📥",
    "Backend 🏭",
    "Quantization 🗜️",
    "Quantization 🗜️ Custom Kernel",
    # primary measurements
    "Prefill (s)",
    "Prefill (s) Custom Kernel",
    "Decode (tokens/s)",
    "Decode (tokens/s) Custom Kernel",
    # speedups
    "Prefill Speedup (%)",
    "Decode Speedup (%)",
]


def get_quant_df(llm_perf_df):
    copy_df = llm_perf_df.copy()
    # seperate vanilla GPTQ experiments from Custom Kernel experiments
    vanilla_df = copy_df[
        (copy_df["Backend 🏭"] == "pytorch")
        & (copy_df["DType 📥"] == "float16")
        & (copy_df["Quantization 🗜️"] == "Unquantized")
    ]
    exllamav1_df = copy_df[(copy_df["Quantization 🗜️"] == "GPTQ.4bit+ExllamaV1")]
    exllamav2_df = copy_df[(copy_df["Quantization 🗜️"] == "GPTQ.4bit+ExllamaV2")]
    gemm_df = copy_df[(copy_df["Quantization 🗜️"] == "AWQ.4bit+GEMM")]
    gemv_df = copy_df[(copy_df["Quantization 🗜️"] == "AWQ.4bit+GEMV")]
    # merge the three dataframes
    exllamav1_df = pd.merge(
        vanilla_df,
        exllamav1_df,
        on=["Model 🤗"],
        suffixes=["", " Custom Kernel"],
    )
    exllamav2_df = pd.merge(
        vanilla_df,
        exllamav2_df,
        on=["Model 🤗"],
        suffixes=["", " Custom Kernel"],
    )
    gemm_df = pd.merge(
        vanilla_df,
        gemm_df,
        on=["Model 🤗"],
        suffixes=["", " Custom Kernel"],
    )
    gemv_df = pd.merge(
        vanilla_df,
        gemv_df,
        on=["Model 🤗"],
        suffixes=["", " Custom Kernel"],
    )
    # concat the two dataframes row-wise
    quant_df = pd.concat([exllamav1_df, exllamav2_df, gemm_df, gemv_df])
    # compute speedups
    quant_df["Prefill Speedup (%)"] = ((quant_df["Prefill (s)"] / quant_df["Prefill (s) Custom Kernel"]) * 100).round(
        2
    ) - 100
    quant_df["Decode Speedup (%)"] = (
        (quant_df["Decode (tokens/s) Custom Kernel"] / quant_df["Decode (tokens/s)"]) * 100
    ).round(2) - 100
    # filter speedups > 1000%
    quant_df = quant_df[quant_df["Prefill Speedup (%)"] < 1000]
    quant_df = quant_df[quant_df["Decode Speedup (%)"] < 1000]

    return quant_df


def get_quant_decode_fig(llm_perf_df):
    quant_df = get_quant_df(llm_perf_df)
    # plot
    decode_fig = px.box(
        quant_df,
        x="Architecture 🏛️",
        y="Decode Speedup (%)",
        color_discrete_sequence=px.colors.qualitative.Light24,
        custom_data=QUANT_DATA,
        color="Quantization 🗜️ Custom Kernel",
        points="all",
    )
    # add hover data
    decode_fig.update_traces(
        hovertemplate="<br>".join([f"<b>{column}:</b> %{{customdata[{i}]}}" for i, column in enumerate(QUANT_DATA)])
    )
    # add layout
    decode_fig.update_layout(
        title={
            "text": "Decode Speedup per Architecture",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="LLM Architecture",
        yaxis_title="Decode Speedup (%)",
        legend_title="Quantization Scheme",
        width=1200,
        height=600,
    )

    return decode_fig


def get_quant_prefill_fig(llm_perf_df):
    quant_df = get_quant_df(llm_perf_df)
    # plot
    prefill_fig = px.box(
        quant_df,
        x="Architecture 🏛️",
        y="Prefill Speedup (%)",
        color_discrete_sequence=px.colors.qualitative.Light24,
        custom_data=QUANT_DATA,
        color="Quantization 🗜️ Custom Kernel",
        points="all",
    )
    # add hover data
    prefill_fig.update_traces(
        hovertemplate="<br>".join([f"<b>{column}:</b> %{{customdata[{i}]}}" for i, column in enumerate(QUANT_DATA)])
    )
    # add layout
    prefill_fig.update_layout(
        title={
            "text": "Prefill Speedup per Architecture",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="LLM Architecture",
        yaxis_title="Prefill Speedup (%)",
        legend_title="Quantization Scheme",
        width=1200,
        height=600,
    )

    return prefill_fig


def create_quant_plots(llm_perf_df):
    # descriptive text
    gr.HTML("👆 Hover over the points 👆 for additional information.", elem_id="text")
    # get figures
    prefill_fig = get_quant_prefill_fig(llm_perf_df)
    decode_fig = get_quant_decode_fig(llm_perf_df)

    # create plots
    prefill_plot = gr.components.Plot(value=prefill_fig, elem_id="plot", show_label=False)
    decode_plot = gr.components.Plot(value=decode_fig, elem_id="plot", show_label=False)

    return prefill_plot, decode_plot
