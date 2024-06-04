from transformers import AutoConfig

LLM_MODEL_ARCHS = {
    "stablelm_epoch": "🔴 StableLM-Epoch",
    "stablelm_alpha": "🔴 StableLM-Alpha",
    "mixformer-sequential": "🧑‍💻 Phi φ",
    "RefinedWebModel": "🦅 Falcon",
    "gpt_bigcode": "⭐ StarCoder",
    "RefinedWeb": "🦅 Falcon",
    "baichuan": "🌊 Baichuan 百川",  # river
    "internlm": "🧑‍🎓 InternLM 书生",  # scholar
    "mistral": "Ⓜ️ Mistral",
    "mixtral": "Ⓜ️ Mixtral",
    "codegen": "♾️ CodeGen",
    "chatglm": "💬 ChatGLM",
    "falcon": "🦅 Falcon",
    "bloom": "🌸 Bloom",
    "llama": "🦙 LLaMA",
    "rwkv": "🐦‍⬛ RWKV",
    "deci": "🔵 deci",
    "Yi": "🫂 Yi 人",  # people
    "mpt": "🧱 MPT",
    # suggest something
    "gpt_neox": "GPT-NeoX",
    "gpt_neo": "GPT-Neo",
    "gpt2": "GPT-2",
    "gptj": "GPT-J",
    "bart": "BART",
}


def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'


def process_architectures(model):
    # return "Unknown"
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        return LLM_MODEL_ARCHS.get(config.model_type, "Unknown")
    except Exception:
        return "Unknown"


def process_score(score, quantization):
    if quantization != "Unquantized":
        return f"{score:.2f}*"
    else:
        return f"{score:.2f} "


def process_quantizations(x):
    if (
        x["config.backend.quantization_scheme"] == "bnb"
        and x["config.backend.quantization_config.load_in_4bit"] is True
    ):
        return "BnB.4bit"
    elif (
        x["config.backend.quantization_scheme"] == "bnb"
        and x["config.backend.quantization_config.load_in_8bit"] is True
    ):
        return "BnB.8bit"
    elif x["config.backend.quantization_scheme"] == "gptq" and x["config.backend.quantization_config.bits"] == 4:
        return "GPTQ.4bit"
    elif x["config.backend.quantization_scheme"] == "awq" and x["config.backend.quantization_config.bits"] == 4:
        return "AWQ.4bit"
    else:
        return "Unquantized"


def process_kernels(x):
    if x["config.backend.quantization_scheme"] == "gptq" and x["config.backend.quantization_config.version"] == 1:
        return "GPTQ.ExllamaV1"

    elif x["config.backend.quantization_scheme"] == "gptq" and x["config.backend.quantization_config.version"] == 2:
        return "GPTQ.ExllamaV2"
    elif (
        x["config.backend.quantization_scheme"] == "awq" and x["config.backend.quantization_config.version"] == "gemm"
    ):
        return "AWQ.GEMM"
    elif (
        x["config.backend.quantization_scheme"] == "awq" and x["config.backend.quantization_config.version"] == "gemv"
    ):
        return "AWQ.GEMV"
    else:
        return "No Kernel"


# def change_tab(query_param):
#     query_param = query_param.replace("'", '"')
#     query_param = json.loads(query_param)

#     if isinstance(query_param, dict) and "tab" in query_param and query_param["tab"] == "plot":
#         return gr.Tabs.update(selected=1)
#     else:
#         return gr.Tabs.update(selected=0)
