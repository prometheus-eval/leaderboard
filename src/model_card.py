import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import huggingface_hub
from huggingface_hub import ModelCard
from huggingface_hub.hf_api import ModelInfo, get_safetensors_metadata
from transformers import AutoConfig, AutoTokenizer


# ht to @Wauplin, thank you for the snippet!
# See https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions/317
def check_model_card(repo_id: str) -> tuple[bool, str]:
    # Returns operation status, and error message
    try:
        card = ModelCard.load(repo_id)
    except huggingface_hub.utils.EntryNotFoundError:
        return False, "Please add a model card to your model to explain how you trained/fine-tuned it.", None

    # Enforce license metadata
    if card.data.license is None:
        if not ("license_name" in card.data and "license_link" in card.data):
            return (
                False,
                (
                    "License not found. Please add a license to your model card using the `license` metadata or a"
                    " `license_name`/`license_link` pair."
                ),
                None,
            )

    # Enforce card content
    if len(card.text) < 200:
        return False, "Please add a description to your model card, it is too short.", None

    return True, "", card


def is_model_on_hub(
    model_name: str, revision: str, token: str = None, trust_remote_code=False, test_tokenizer=False
) -> tuple[bool, str, AutoConfig]:
    try:
        config = AutoConfig.from_pretrained(
            model_name, revision=revision, trust_remote_code=trust_remote_code, token=token
        )  # , force_download=True)
        if test_tokenizer:
            try:
                tk = AutoTokenizer.from_pretrained(
                    model_name, revision=revision, trust_remote_code=trust_remote_code, token=token
                )
            except ValueError as e:
                return (False, f"uses a tokenizer which is not in a transformers release: {e}", None)
            except Exception:
                return (
                    False,
                    "'s tokenizer cannot be loaded. Is your tokenizer class in a stable transformers release, and correctly configured?",
                    None,
                )
        return True, None, config

    except ValueError:
        return (
            False,
            "needs to be launched with `trust_remote_code=True`. For safety reason, we do not allow these models to be automatically submitted to the leaderboard.",
            None,
        )

    except Exception as e:
        if "You are trying to access a gated repo." in str(e):
            return True, "uses a gated model.", None
        return False, f"was not found or misconfigured on the hub! Error raised was {e.args[0]}", None


def get_model_size(model_info: ModelInfo, precision: str):
    size_pattern = re.compile(r"(\d+\.)?\d+(b|m)")
    safetensors = None
    try:
        safetensors = get_safetensors_metadata(model_info.id)
    except Exception as e:
        print(e)

    if safetensors is not None:
        model_size = round(sum(safetensors.parameter_count.values()) / 1e9, 3)
    else:
        try:
            size_match = re.search(size_pattern, model_info.id.lower())
            model_size = size_match.group(0)
            model_size = round(float(model_size[:-1]) if model_size[-1] == "b" else float(model_size[:-1]) / 1e3, 3)
        except AttributeError:
            return 0  # Unknown model sizes are indicated as 0, see NUMERIC_INTERVALS in app.py

    size_factor = 8 if (precision == "GPTQ" or "gptq" in model_info.id.lower()) else 1
    model_size = size_factor * model_size
    return model_size


def get_model_arch(model_info: ModelInfo):
    return model_info.config.get("architectures", "Unknown")


def get_model_tags(model_card, model: str):
    is_merge_from_metadata = False
    is_moe_from_metadata = False

    tags = []
    if model_card is None:
        return tags
    if model_card.data.tags:
        is_merge_from_metadata = any(
            [tag in model_card.data.tags for tag in ["merge", "moerge", "mergekit", "lazymergekit"]]
        )
        is_moe_from_metadata = any([tag in model_card.data.tags for tag in ["moe", "moerge"]])

    is_merge_from_model_card = any(
        keyword in model_card.text.lower() for keyword in ["merged model", "merge model", "moerge"]
    )
    if is_merge_from_model_card or is_merge_from_metadata:
        tags.append("merge")
    is_moe_from_model_card = any(keyword in model_card.text.lower() for keyword in ["moe", "mixtral"])
    # Hardcoding because of gating problem
    if "Qwen/Qwen1.5-32B" in model:
        is_moe_from_model_card = False
    is_moe_from_name = "moe" in model.lower().replace("/", "-").replace("_", "-").split("-")
    if is_moe_from_model_card or is_moe_from_name or is_moe_from_metadata:
        tags.append("moe")

    return tags


def test():
    model = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Test check_model_card
    status, error, card = check_model_card(model)

    # Test is_model_on_hub
    status2, error2, config2 = is_model_on_hub(model, "main")
    assert status == True
    print(status2, error2, config2)

    # Test get_model_size
    model_info = ModelInfo(id=model)
    precision = "GPTQ"
    model_size = get_model_size(model_info, precision)
    print(model_size)

    import pdb

    pdb.set_trace()

    # Test get_model_arch
    # model_arch = get_model_arch(model_info)

    pass


if __name__ == "__main__":
    test()
