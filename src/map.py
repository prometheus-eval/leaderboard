import gradio as gr
import plotly.express as px

SCORE_MEMORY_LATENCY_DATA = [
    "Model 🤗",
    "Backend 🏭",
    "Precision 📥",
    "Params (B)",
    "Quantization 🗜️",
    "Attention 👁️",
    "Kernel ⚛️",
    "Open LLM Score (%)",
    "Prefill (s)",
    "Decode (tokens/s)",
    "Memory (MB)",
    "End-to-End (s)",
    "Architecture 🏛️",
]


def get_lat_score_mem_fig(llm_perf_df):
    copy_df = llm_perf_df.copy()
    # plot
    # filter nan memory
    fig = px.scatter(
        copy_df,
        size="Memory (MB)",
        x="End-to-End (s)",
        y="Open LLM Score (%)",
        color="Architecture 🏛️",
        custom_data=SCORE_MEMORY_LATENCY_DATA,
        color_discrete_sequence=px.colors.qualitative.Light24,
    )
    fig.update_traces(
        hovertemplate="<br>".join(
            [f"<b>{column}:</b> %{{customdata[{i}]}}" for i, column in enumerate(SCORE_MEMORY_LATENCY_DATA)]
        )
    )
    fig.update_layout(
        title={
            "text": "Latency vs. Score vs. Memory",
            "xanchor": "center",
            "yanchor": "top",
            "y": 0.95,
            "x": 0.5,
        },
        xaxis_title="Time To Generate 64 Tokens (s)",
        yaxis_title="Open LLM Score (%)",
        legend_title="LLM Architecture",
        width=1200,
        height=600,
    )
    # update x range with 95 percentile of
    fig.update_xaxes(range=[-0.5, copy_df["End-to-End (s)"].quantile(0.90)])

    return fig


def create_lat_score_mem_plot(llm_perf_df):
    # descriptive text
    gr.HTML("👆 Hover over the points 👆 for additional information. ", elem_id="text")
    gr.HTML("📊 We only show the top 90% LLMs based on latency ⌛", elem_id="text")
    # get figure
    fig = get_lat_score_mem_fig(llm_perf_df)
    # create plot
    plot = gr.components.Plot(
        value=fig,
        elem_id="plot",
        show_label=False,
    )

    return plot
