import gradio as gr
import pandas as pd
import plotly.express as px

ATTN_DATA = [
    # open llm
    "Model 🤗",
    "Experiment 🧪",
    "Params (B)",
    "Architecture 🏛️",
    "Open LLM Score (%)",
    # deployment settings
    "Backend 🏭",
    "Quantization 🗜️",
    "Precision 📥",
    "Attention 👁️",
    "Kernel ⚛️",
    # primary measurements
    "Prefill (s)",
    "Decode (tokens/s)",
    # speedups
    "Prefill Speedup (%)",
    "Decode Speedup (%)",
]


def get_attn_df(open_llm_perf_df):
    copy_df = open_llm_perf_df.copy()
    copy_df["Quantization & Kernel"] = copy_df["Quantization 🗜️"] + " & " + copy_df["Kernel ⚛️"]

    eager_df = copy_df[(copy_df["Attention 👁️"] == "Eager")]
    sdpa_df = copy_df[(copy_df["Attention 👁️"] == "SDPA")]
    fa2_df = copy_df[(copy_df["Attention 👁️"] == "FAv2")]

    sdpa_df = pd.merge(
        eager_df,
        sdpa_df,
        on=["Model 🤗", "Quantization & Kernel"],
        suffixes=["", " other"],
    )
    fa2_df = pd.merge(
        eager_df,
        fa2_df,
        on=["Model 🤗", "Quantization & Kernel"],
        suffixes=["", " other"],
    )

    attn_df = pd.concat([sdpa_df, fa2_df])

    # compute speedups
    attn_df["Prefill Speedup (%)"] = ((attn_df["Prefill (s)"] / attn_df["Prefill (s) other"]) * 100).round(2) - 100
    attn_df["Decode Speedup (%)"] = ((attn_df["Decode (tokens/s) other"] / attn_df["Decode (tokens/s)"]) * 100).round(
        2
    ) - 100

    return attn_df


def get_attn_prefill_fig(open_llm_perf_df):
    attn_df = get_attn_df(open_llm_perf_df)
    # plot
    prefill_fig = px.box(
        attn_df,
        x="Architecture 🏛️",
        y="Prefill Speedup (%)",
        color_discrete_sequence=px.colors.qualitative.Light24,
        custom_data=ATTN_DATA,
        color="Attention 👁️ other",
        points="all",
    )
    # add hover data
    prefill_fig.update_traces(
        hovertemplate="<br>".join([f"<b>{column}:</b> %{{customdata[{i}]}}" for i, column in enumerate(ATTN_DATA)])
    )
    # add layout
    prefill_fig.update_layout(
        title={
            "text": "Prefill Speedup per Architecture, Compared To Eager Attention",
            "xanchor": "center",
            "yanchor": "top",
            "y": 0.95,
            "x": 0.5,
        },
        yaxis_title="Prefill Speedup (%)",
        xaxis_title="LLM Architecture",
        legend_title="Attention",
        width=1200,
        height=600,
    )

    return prefill_fig


def get_attn_decode_fig(open_llm_perf_df):
    attn_df = get_attn_df(open_llm_perf_df)
    print(len(attn_df))
    # plot
    decode_fig = px.box(
        attn_df,
        x="Architecture 🏛️",
        y="Decode Speedup (%)",
        color_discrete_sequence=px.colors.qualitative.Light24,
        custom_data=ATTN_DATA,
        color="Attention 👁️ other",
        points="all",
    )
    # add hover data
    decode_fig.update_traces(
        hovertemplate="<br>".join([f"<b>{column}:</b> %{{customdata[{i}]}}" for i, column in enumerate(ATTN_DATA)])
    )
    # add layout
    decode_fig.update_layout(
        title={
            "text": "Decode Speedup per Architecture, Compared To Eager Attention",
            "xanchor": "center",
            "yanchor": "top",
            "y": 0.95,
            "x": 0.5,
        },
        yaxis_title="Decode Speedup (%)",
        xaxis_title="LLM Architecture",
        legend_title="Attention",
        width=1200,
        height=600,
    )

    return decode_fig


def create_attn_plots(open_llm_perf_df):
    # descriptive text
    gr.HTML("👆 Hover over the points 👆 for additional information.", elem_id="text")
    # get figures
    prefill_fig = get_attn_prefill_fig(open_llm_perf_df)
    decode_fig = get_attn_decode_fig(open_llm_perf_df)

    # create plots
    prefill_plot = gr.components.Plot(value=prefill_fig, elem_id="plot", show_label=False)
    decode_plot = gr.components.Plot(value=decode_fig, elem_id="plot", show_label=False)

    return prefill_plot, decode_plot
