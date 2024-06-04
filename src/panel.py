import gradio as gr

from src.leaderboard import BGB_COLUMN_MAPPING, get_bgb_leaderboard_df, get_leaderboard_df
from src.llm_perf import get_eval_df, get_llm_perf_df


def select_columns_fn(machine, columns, search, llm_perf_df=None):
    if llm_perf_df is None:
        llm_perf_df = get_llm_perf_df(machine=machine)

    selected_leaderboard_df = get_leaderboard_df(llm_perf_df)
    selected_leaderboard_df = selected_leaderboard_df[
        selected_leaderboard_df["Model 🤗"].str.contains(search, case=False)
    ]
    selected_leaderboard_df = selected_leaderboard_df[columns]

    return selected_leaderboard_df


def select_columns_bgb_fn(machine, columns, search, type_checkboxes, param_slider, eval_df=None):
    if eval_df is None:
        eval_df = get_eval_df(machine)

    selected_leaderboard_df = get_bgb_leaderboard_df(eval_df)
    selected_leaderboard_df = selected_leaderboard_df[
        selected_leaderboard_df["Model 🤗"].str.contains(search, case=False)
    ]

    print(param_slider)

    import pdb

    pdb.set_trace()

    columns = ["Model 🤗"] + columns + type_checkboxes

    return selected_leaderboard_df[columns]


def create_select_callback(
    # fixed
    machine_textbox,
    # interactive
    columns_checkboxes,
    search_bar,
    type_checkboxes,
    param_slider,
    # outputs
    leaderboard_table,
):
    columns_checkboxes.change(
        fn=select_columns_bgb_fn,
        inputs=[machine_textbox, columns_checkboxes, search_bar, type_checkboxes, param_slider],
        outputs=[leaderboard_table],
    )
    search_bar.change(
        fn=select_columns_bgb_fn,
        inputs=[machine_textbox, columns_checkboxes, search_bar, type_checkboxes, param_slider],
        outputs=[leaderboard_table],
    )
