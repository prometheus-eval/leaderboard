import gradio as gr

from src.leaderboard import get_leaderboard_df
from src.llm_perf import get_llm_perf_df

# from attention_implementations import get_attn_decode_fig, get_attn_prefill_fig
# from custom_kernels import get_kernel_decode_fig, get_kernel_prefill_fig
from src.map import get_lat_score_mem_fig


def create_control_panel(machine: str):
    # controls
    machine_textbox = gr.Textbox(value=machine, visible=False)
    with gr.Accordion("Control Panel 🎛️", open=False, elem_id="control-panel"):
        with gr.Row():
            with gr.Column(scale=2, variant="panel"):
                score_slider = gr.Slider(
                    label="Open LLM Score (%) 📈",
                    info="🎚️ Slide to minimum Open LLM score",
                    value=0,
                    elem_id="threshold-slider",
                )
            with gr.Column(scale=2, variant="panel"):
                memory_slider = gr.Slider(
                    label="Peak Memory (MB) 📈",
                    info="🎚️ Slide to maximum Peak Memory",
                    minimum=0,
                    maximum=80 * 1024,
                    value=80 * 1024,
                    elem_id="memory-slider",
                )
            with gr.Column(scale=1, variant="panel"):
                backend_checkboxes = gr.CheckboxGroup(
                    label="Backends 🏭",
                    choices=["pytorch"],
                    value=["pytorch"],
                    info="☑️ Select the backends",
                    elem_id="backend-checkboxes",
                )
        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                datatype_checkboxes = gr.CheckboxGroup(
                    label="Precision 📥",
                    choices=["float32", "float16", "bfloat16"],
                    value=["float32", "float16", "bfloat16"],
                    info="☑️ Select the load data types",
                    elem_id="dtype-checkboxes",
                )
            with gr.Column(scale=1, variant="panel"):
                optimization_checkboxes = gr.CheckboxGroup(
                    label="Attentions 👁️",
                    choices=["Eager", "SDPA", "FAv2"],
                    value=["Eager", "SDPA", "FAv2"],
                    info="☑️ Select the optimization",
                    elem_id="optimization-checkboxes",
                )
        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                quantization_checkboxes = gr.CheckboxGroup(
                    label="Quantizations 🗜️",
                    choices=[
                        "Unquantized",
                        "BnB.4bit",
                        "BnB.8bit",
                        "AWQ.4bit",
                        "GPTQ.4bit",
                    ],
                    value=[
                        "Unquantized",
                        "BnB.4bit",
                        "BnB.8bit",
                        "AWQ.4bit",
                        "GPTQ.4bit",
                    ],
                    info="☑️ Select the quantization schemes",
                    elem_id="quantization-checkboxes",
                    elem_classes="boxed-option",
                )
            with gr.Column(scale=1, variant="panel"):
                kernels_checkboxes = gr.CheckboxGroup(
                    label="Kernels ⚛️",
                    choices=[
                        "No Kernel",
                        "GPTQ.ExllamaV1",
                        "GPTQ.ExllamaV2",
                        "AWQ.GEMM",
                        "AWQ.GEMV",
                    ],
                    value=[
                        "No Kernel",
                        "GPTQ.ExllamaV1",
                        "GPTQ.ExllamaV2",
                        "AWQ.GEMM",
                        "AWQ.GEMV",
                    ],
                    info="☑️ Select the custom kernels",
                    elem_id="kernel-checkboxes",
                    elem_classes="boxed-option",
                )
        with gr.Row():
            filter_button = gr.Button(
                value="Filter 🚀",
                elem_id="filter-button",
                elem_classes="boxed-option",
            )

    return (
        filter_button,
        machine_textbox,
        score_slider,
        memory_slider,
        backend_checkboxes,
        datatype_checkboxes,
        optimization_checkboxes,
        quantization_checkboxes,
        kernels_checkboxes,
    )


def filter_rows_fn(
    machine,
    # inputs
    score,
    memory,
    backends,
    precisions,
    attentions,
    quantizations,
    kernels,
    # interactive
    columns,
    search,
):
    llm_perf_df = get_llm_perf_df(machine=machine)
    # print(attentions)
    # print(llm_perf_df["Attention 👁️"].unique())
    filtered_llm_perf_df = llm_perf_df[
        llm_perf_df["Model 🤗"].str.contains(search, case=False)
        & llm_perf_df["Backend 🏭"].isin(backends)
        & llm_perf_df["Precision 📥"].isin(precisions)
        & llm_perf_df["Attention 👁️"].isin(attentions)
        & llm_perf_df["Quantization 🗜️"].isin(quantizations)
        & llm_perf_df["Kernel ⚛️"].isin(kernels)
        & (llm_perf_df["Open LLM Score (%)"] >= score)
        & (llm_perf_df["Memory (MB)"] <= memory)
    ]
    selected_filtered_llm_perf_df = select_columns_fn(machine, columns, search, filtered_llm_perf_df)
    selected_filtered_lat_score_mem_fig = get_lat_score_mem_fig(filtered_llm_perf_df)
    # filtered_bt_prefill_fig = get_bt_prefill_fig(filtered_df)
    # filtered_bt_decode_fig = get_bt_decode_fig(filtered_df)
    # filtered_fa2_prefill_fig = get_fa2_prefill_fig(filtered_df)
    # filtered_fa2_decode_fig = get_fa2_decode_fig(filtered_df)
    # filtered_quant_prefill_fig = get_quant_prefill_fig(filtered_df)
    # filtered_quant_decode_fig = get_quant_decode_fig(filtered_df)

    return [
        selected_filtered_llm_perf_df,
        selected_filtered_lat_score_mem_fig,
        # filtered_bt_prefill_fig,
        # filtered_bt_decode_fig,
        # filtered_fa2_prefill_fig,
        # filtered_fa2_decode_fig,
        # filtered_quant_prefill_fig,
        # filtered_quant_decode_fig,
    ]


def create_control_callback(
    # button
    filter_button,
    # fixed
    machine_textbox,
    # inputs
    score_slider,
    memory_slider,
    backend_checkboxes,
    datatype_checkboxes,
    optimization_checkboxes,
    quantization_checkboxes,
    kernels_checkboxes,
    # interactive
    columns_checkboxes,
    search_bar,
    # outputs
    leaderboard_table,
    # lat_score_mem_plot,
    # attn_prefill_plot,
    # attn_decode_plot,
    # fa2_prefill_plot,
    # fa2_decode_plot,
    # quant_prefill_plot,
    # quant_decode_plot,
):
    filter_button.click(
        fn=filter_rows_fn,
        inputs=[
            # fixed
            machine_textbox,
            # inputs
            score_slider,
            memory_slider,
            backend_checkboxes,
            datatype_checkboxes,
            optimization_checkboxes,
            quantization_checkboxes,
            kernels_checkboxes,
            # interactive
            columns_checkboxes,
            search_bar,
        ],
        outputs=[
            leaderboard_table,
            # lat_score_mem_plot,
            # attn_prefill_plot,
            # attn_decode_plot,
            # fa2_prefill_plot,
            # fa2_decode_plot,
            # quant_prefill_plot,
            # quant_decode_plot,
        ],
    )


def select_columns_fn(machine, columns, search, llm_perf_df=None):
    if llm_perf_df is None:
        llm_perf_df = get_llm_perf_df(machine=machine)

    selected_leaderboard_df = get_leaderboard_df(llm_perf_df)
    selected_leaderboard_df = selected_leaderboard_df[
        selected_leaderboard_df["Model 🤗"].str.contains(search, case=False)
    ]
    selected_leaderboard_df = selected_leaderboard_df[columns]

    return selected_leaderboard_df


def create_select_callback(
    # fixed
    machine_textbox,
    # interactive
    columns_checkboxes,
    search_bar,
    # outputs
    leaderboard_table,
):
    columns_checkboxes.change(
        fn=select_columns_fn,
        inputs=[machine_textbox, columns_checkboxes, search_bar],
        outputs=[leaderboard_table],
    )
    search_bar.change(
        fn=select_columns_fn,
        inputs=[machine_textbox, columns_checkboxes, search_bar],
        outputs=[leaderboard_table],
    )
