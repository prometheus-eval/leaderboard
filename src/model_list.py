MODELS = {
    "pretrained": {
        "<=4B": [
            "microsoft/phi-1",
            "microsoft/phi-1_5",
            "microsoft/phi-2",
            "Qwen/Qwen1.5-0.5B",
            "Qwen/Qwen1.5-1.8B",
            "Qwen/Qwen1.5-4B",
            "google/gemma-2b",
            "allenai/OLMo-1B",
        ],
        "<=7B": [
            "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1",
            "Qwen/Qwen1.5-7B",
            "01-ai/Yi-6B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-7b-hf",
            "EleutherAI/llemma_7b",
            "allenai/OLMo-7B",
            "mistral-community/Mistral-7B-v0.2",
        ],
        "<=14B": [
            "Qwen/Qwen1.5-14B",
            "meta-llama/Llama-2-13b-hf",
            "codellama/CodeLlama-13b-hf",
            "upstage/SOLAR-10.7B-v1.0",
            "meta-llama/Meta-Llama-3-8B",
        ],
        "<=50B": [
            "01-ai/Yi-34B",
            "EleutherAI/llemma_34b",
            "codellama/CodeLlama-34b-hf",
            "mistralai/Mixtral-8x7B-v0.1",
            "Qwen/Qwen1.5-32B",
        ],
        "<=75B": [
            "meta-llama/Llama-2-70b-hf",
            "codellama/CodeLlama-70b-hf",
            "meta-llama/Meta-Llama-3-70B",
            "Qwen/Qwen1.5-72B",
        ],
        "<=175B": [
            "mistral-community/Mixtral-8x22B-v0.1-AWQ",
        ],
    },
    "instruction_tuned": {
        "<=4B": [
            "Qwen/Qwen1.5-0.5B-Chat",
            "Qwen/Qwen1.5-1.8B-Chat",
            "Qwen/Qwen1.5-4B-Chat",
            "google/gemma-2b-it",
            "google/gemma-1.1-2b-it",
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-mini-128k-instruct",
        ],
        "<=7B": [
            "google/gemma-7b-it",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "Qwen/Qwen1.5-7B-Chat",
            "01-ai/Yi-6B-Chat",
            "meta-llama/Llama-2-7b-chat-hf",
            "codellama/CodeLlama-7b-Instruct-hf",
            "allenai/OLMo-7B-SFT",
            "allenai/OLMo-7B-Instruct",
            "allenai/tulu-2-7b",
            "allenai/tulu-2-dpo-7b",
            "allenai/codetulu-2-7b",
            "microsoft/Orca-2-7b",
            "openchat/openchat-3.5-0106",
            "teknium/OpenHermes-2-Mistral-7B",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
            "HuggingFaceH4/zephyr-7b-beta",
            "berkeley-nest/Starling-LM-7B-alpha",
            "Nexusflow/Starling-LM-7B-beta",
            "kaist-ai/mistral-orpo-alpha",
            "kaist-ai/mistral-orpo-beta",
            "google/gemma-1.1-7b-it",
        ],
        "<=14B": [
            "Qwen/Qwen1.5-14B-Chat",
            "meta-llama/Llama-2-13b-chat-hf",
            "codellama/CodeLlama-13b-Instruct-hf",
            "allenai/tulu-2-13b",
            "allenai/tulu-2-dpo-13b",
            "allenai/codetulu-2-13b",
            "microsoft/Orca-2-13b",
            "upstage/SOLAR-10.7B-Instruct-v1.0",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "CohereForAI/aya-101",
        ],
        "<=50B": [
            "01-ai/Yi-34B-Chat",
            "codellama/CodeLlama-34b-Instruct-hf",
            "allenai/codetulu-2-34b",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "NousResearch/Nous-Hermes-2-Yi-34B",
            "CohereForAI/c4ai-command-r-v01",
            "Qwen/Qwen1.5-32B-Chat",
        ],
        "<=75B": [
            "meta-llama/Llama-2-70b-chat-hf",
            "codellama/CodeLlama-70b-Instruct-hf",
            "Qwen/Qwen1.5-72B-Chat",
            "allenai/tulu-2-dpo-70b",
            "meta-llama/Meta-Llama-3-70B-Instruct",
        ],
        "<=175B": [
            "alpindale/c4ai-command-r-plus-GPTQ",
            "MaziyarPanahi/zephyr-orpo-141b-A35b-v0.1-AWQ",
            "MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ",
        ],
    },
}

API_MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "mistral-large",
    "mistral-medium",
    "gemini-1.0-pro",
    "gemini-pro-1.5",
    "google/gemini-flash-1.5",
    "qwen/qwen-110b-chat",
]


ORDERED_MODELS = [
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B",
    "google/gemma-2b",
    "allenai/OLMo-1B",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "google/gemma-2b-it",
    "google/gemma-1.1-2b-it",
    "google/gemma-7b",
    "mistralai/Mistral-7B-v0.1",
    "mistral-community/Mistral-7B-v0.2",
    "Qwen/Qwen1.5-7B",
    "01-ai/Yi-6B",
    "meta-llama/Llama-2-7b-hf",
    "codellama/CodeLlama-7b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "EleutherAI/llemma_7b",
    "allenai/OLMo-7B",
    "google/gemma-7b-it",
    "google/gemma-1.1-7b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen1.5-7B-Chat",
    "01-ai/Yi-6B-Chat",
    "meta-llama/Llama-2-7b-chat-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "allenai/OLMo-7B-SFT",
    "allenai/OLMo-7B-Instruct",
    "allenai/tulu-2-7b",
    "allenai/tulu-2-dpo-7b",
    "allenai/codetulu-2-7b",
    "microsoft/Orca-2-7b",
    "openchat/openchat-3.5-0106",
    "teknium/OpenHermes-2-Mistral-7B",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "Starling-LM-7B-alpha",
    "Starling-LM-7B-beta",
    "kaist-ai/mistral-orpo-alpha",
    "kaist-ai/mistral-orpo-beta",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen1.5-14B",
    "meta-llama/Llama-2-13b-hf",
    "codellama/CodeLlama-13b-hf",
    "upstage/SOLAR-10.7B-v1.0",
    "Qwen/Qwen1.5-14B-Chat",
    "upstage/SOLAR-10.7B-Instruct-v1.0",
    "CohereForAI/aya-101",
    "meta-llama/Llama-2-13b-chat-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "allenai/tulu-2-13b",
    "allenai/tulu-2-dpo-13b",
    "allenai/codetulu-2-13b",
    "microsoft/Orca-2-13b",
    "01-ai/Yi-34B",
    "EleutherAI/llemma_34b",
    "Qwen/Qwen1.5-32B",
    "codellama/CodeLlama-34b-hf",
    "mistralai/Mixtral-8x7B-v0.1",
    "01-ai/Yi-34B-Chat",
    "NousResearch/Nous-Hermes-2-Yi-34B",
    "codellama/CodeLlama-34b-Instruct-hf",
    "allenai/codetulu-2-34b",
    "Qwen/Qwen1.5-32B-Chat",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "CohereForAI/c4ai-command-r-v01",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-70b-hf",
    "mistral-community/Mixtral-8x22B-v0.1-AWQ",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen1.5-72B",
    "meta-llama/Llama-2-70b-chat-hf",
    "codellama/CodeLlama-70b-Instruct-hf",
    "allenai/tulu-2-dpo-70b",
    "alpindale/c4ai-command-r-plus-GPTQ",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ",
    "MaziyarPanahi/zephyr-orpo-141b-A35b-v0.1-AWQ",
    "Qwen/Qwen1.5-72B-Chat",
    "qwen/qwen-110b-chat",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o-2024-05-13",
    "mistral-medium",
    "mistral-large",
    "gemini-1.0-pro",
    "gemini-pro-1.5",
    "google/gemini-flash-1.5",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
]


bgb_trained_models = [
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B",
    "google/gemma-2b",
    "allenai/OLMo-1B",
    "google/gemma-7b",
    "mistralai/Mistral-7B-v0.1",
    "Qwen/Qwen1.5-7B",
    "01-ai/Yi-6B",
    "meta-llama/Llama-2-7b-hf",
    "codellama/CodeLlama-7b-hf",
    "EleutherAI/llemma_7b",
    "allenai/OLMo-7B",
    "Qwen/Qwen1.5-14B",
    "meta-llama/Llama-2-13b-hf",
    "codellama/CodeLlama-13b-hf",
    "upstage/SOLAR-10.7B-v1.0",
    "01-ai/Yi-34B",
    "EleutherAI/llemma_34b",
    "codellama/CodeLlama-34b-hf",
    "mistralai/Mixtral-8x7B-v0.1",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-70b-hf",
    "Qwen/Qwen1.5-72B",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen1.5-7B-Chat",
    "01-ai/Yi-6B-Chat",
    "meta-llama/Llama-2-7b-chat-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "allenai/OLMo-7B-SFT",
    "allenai/OLMo-7B-Instruct",
    "allenai/tulu-2-7b",
    "allenai/tulu-2-dpo-7b",
    "allenai/codetulu-2-7b",
    "microsoft/Orca-2-7b",
    "openchat/openchat-3.5-0106",
    "teknium/OpenHermes-2-Mistral-7B",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen1.5-14B-Chat",
    "meta-llama/Llama-2-13b-chat-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "allenai/tulu-2-13b",
    "allenai/tulu-2-dpo-13b",
    "allenai/codetulu-2-13b",
    "microsoft/Orca-2-13b",
    "01-ai/Yi-34B-Chat",
    "codellama/CodeLlama-34b-Instruct-hf",
    "allenai/codetulu-2-34b",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "NousResearch/Nous-Hermes-2-Mistral-8x7B-SFT",
    "NousResearch/Nous-Hermes-2-Mistral-8x7B-DPO",
    "NousResearch/Nous-Hermes-2-Yi-34B",
    "meta-llama/Llama-2-70b-chat-hf",
    "codellama/CodeLlama-70b-Instruct-hf",
    "Qwen/Qwen1.5-72B-Chat",
    "allenai/tulu-2-dpo-72b",
]


MODEL_MAPPING = {
    "microsoft/phi-1": [1.3, "Base"],
    "microsoft/phi-1_5": [1.3, "Base"],
    "microsoft/phi-2": [2.7, "Base"],
    "Qwen/Qwen1.5-0.5B": [0.5, "Base"],
    "Qwen/Qwen1.5-1.8B": [1.8, "Base"],
    "Qwen/Qwen1.5-4B": [4.0, "Base"],
    "google/gemma-2b": [2.0, "Base"],
    "allenai/OLMo-1B": [1.0, "Base"],
    "Qwen/Qwen1.5-0.5B-Chat": [0.5, "Chat", "Qwen/Qwen1.5-0.5B"],
    "Qwen/Qwen1.5-1.8B-Chat": [1.8, "Chat", "Qwen/Qwen1.5-1.8B"],
    "Qwen/Qwen1.5-4B-Chat": [4.0, "Chat", "Qwen/Qwen1.5-4B"],
    "microsoft/Phi-3-mini-4k-instruct": [3.8, "Chat"],
    "microsoft/Phi-3-mini-128k-instruct": [3.8, "Chat"],
    "google/gemma-2b-it": [2.0, "Chat", "google/gemma-2b"],
    "google/gemma-1.1-2b-it": [2.0, "Chat"],
    "google/gemma-7b": [7.0, "Base"],
    "mistralai/Mistral-7B-v0.1": [7.0, "Base"],
    "mistral-community/Mistral-7B-v0.2": [7.0, "Base"],
    "Qwen/Qwen1.5-7B": [7.0, "Base"],
    "01-ai/Yi-6B": [6.0, "Base"],
    "meta-llama/Llama-2-7b-hf": [7.0, "Base"],
    "codellama/CodeLlama-7b-hf": [7.0, "Base"],
    "meta-llama/Meta-Llama-3-8B": [8.0, "Base"],
    "EleutherAI/llemma_7b": [7.0, "Base"],
    "allenai/OLMo-7B": [7.0, "Base"],
    "google/gemma-7b-it": [7.0, "Chat", "google/gemma-7b"],
    "google/gemma-1.1-7b-it": [7.0, "Chat"],
    "mistralai/Mistral-7B-Instruct-v0.2": [7.0, "Chat", "mistral-community/Mistral-7B-v0.2"],
    "Qwen/Qwen1.5-7B-Chat": [7.0, "Chat", "Qwen/Qwen1.5-7B"],
    "01-ai/Yi-6B-Chat": [6.0, "Chat", "01-ai/Yi-6B"],
    "meta-llama/Llama-2-7b-chat-hf": [7.0, "Chat", "meta-llama/Llama-2-7b-hf"],
    "codellama/CodeLlama-7b-Instruct-hf": [7.0, "Chat", "codellama/CodeLlama-7b-hf"],
    "meta-llama/Meta-Llama-3-8B-Instruct": [8.0, "Chat", "meta-llama/Meta-Llama-3-8B"],
    "allenai/OLMo-7B-SFT": [7.0, "Chat", "allenai/OLMo-7B"],
    "allenai/OLMo-7B-Instruct": [7.0, "Chat", "allenai/OLMo-7B"],
    "allenai/tulu-2-7b": [7.0, "Chat", "meta-llama/Llama-2-7b-hf"],
    "allenai/tulu-2-dpo-7b": [7.0, "Chat", "meta-llama/Llama-2-7b-hf"],
    "allenai/codetulu-2-7b": [7.0, "Chat", "codellama/CodeLlama-7b-hf"],
    "microsoft/Orca-2-7b": [7.0, "Chat", "meta-llama/Llama-2-7b-hf"],
    "openchat/openchat-3.5-0106": [7.0, "Chat", "mistralai/Mistral-7B-v0.1"],
    "teknium/OpenHermes-2-Mistral-7B": [7.0, "Chat", "mistralai/Mistral-7B-v0.1"],
    "teknium/OpenHermes-2.5-Mistral-7B": [7.0, "Chat", "mistralai/Mistral-7B-v0.1"],
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO": [7.0, "Chat", "mistralai/Mistral-7B-v0.1"],
    "Starling-LM-7B-alpha": [7.0, "Chat"],
    "Starling-LM-7B-beta": [7.0, "Chat"],
    "kaist-ai/mistral-orpo-alpha": [7.0, "Chat", "mistralai/Mistral-7B-v0.1"],
    "kaist-ai/mistral-orpo-beta": [7.0, "Chat", "mistralai/Mistral-7B-v0.1"],
    "HuggingFaceH4/zephyr-7b-beta": [7.0, "Chat", "mistralai/Mistral-7B-v0.1"],
    "Qwen/Qwen1.5-14B": [14.0, "Base"],
    "meta-llama/Llama-2-13b-hf": [13.0, "Base"],
    "codellama/CodeLlama-13b-hf": [13.0, "Base"],
    "upstage/SOLAR-10.7B-v1.0": [10.7, "Base"],
    "Qwen/Qwen1.5-14B-Chat": [14.0, "Chat", "Qwen/Qwen1.5-14B"],
    "upstage/SOLAR-10.7B-Instruct-v1.0": [10.7, "Chat", "upstage/SOLAR-10.7B-v1.0"],
    "CohereForAI/aya-101": [13.0, "Chat"],
    "meta-llama/Llama-2-13b-chat-hf": [13.0, "Chat", "meta-llama/Llama-2-13b-hf"],
    "codellama/CodeLlama-13b-Instruct-hf": [13.0, "Chat", "codellama/CodeLlama-13b-hf"],
    "allenai/tulu-2-13b": [13.0, "Chat", "meta-llama/Llama-2-13b-hf"],
    "allenai/tulu-2-dpo-13b": [13.0, "Chat", "meta-llama/Llama-2-13b-hf"],
    "allenai/codetulu-2-13b": [13.0, "Chat", "codellama/CodeLlama-13b-hf"],
    "microsoft/Orca-2-13b": [13.0, "Chat", "meta-llama/Llama-2-13b-hf"],
    "01-ai/Yi-34B": [34.0, "Base"],
    "EleutherAI/llemma_34b": [34.0, "Base"],
    "Qwen/Qwen1.5-32B": [32.0, "Base"],
    "codellama/CodeLlama-34b-hf": [34.0, "Base"],
    "mistralai/Mixtral-8x7B-v0.1": [46.7, "Base"],
    "01-ai/Yi-34B-Chat": [34.0, "Chat", "01-ai/Yi-34B"],
    "NousResearch/Nous-Hermes-2-Yi-34B": [34.0, "Chat", "01-ai/Yi-34B"],
    "codellama/CodeLlama-34b-Instruct-hf": [34.0, "Chat", "codellama/CodeLlama-34b-hf"],
    "allenai/codetulu-2-34b": [34.0, "Chat", "codellama/CodeLlama-34b-hf"],
    "Qwen/Qwen1.5-32B-Chat": [32.0, "Chat", "Qwen/Qwen1.5-32B"],
    "mistralai/Mixtral-8x7B-Instruct-v0.1": [46.7, "Chat", "mistralai/Mixtral-8x7B-v0.1"],
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT": [46.7, "Chat", "mistralai/Mixtral-8x7B-v0.1"],
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": [46.7, "Chat", "mistralai/Mixtral-8x7B-v0.1"],
    "CohereForAI/c4ai-command-r-v01": [35.0, "Chat"],
    "meta-llama/Llama-2-70b-hf": [70.0, "Base"],
    "codellama/CodeLlama-70b-hf": [70.0, "Base"],
    "mistral-community/Mixtral-8x22B-v0.1-AWQ": ["AWQ", "Base"],
    "meta-llama/Meta-Llama-3-70B": [70.0, "Base"],
    "Qwen/Qwen1.5-72B": [72.0, "Base"],
    "meta-llama/Llama-2-70b-chat-hf": [70.0, "Chat", "meta-llama/Llama-2-70b-hf"],
    "codellama/CodeLlama-70b-Instruct-hf": [70.0, "Chat", "codellama/CodeLlama-70b-hf"],
    "allenai/tulu-2-dpo-70b": [70.0, "Chat", "meta-llama/Llama-2-70b-hf"],
    "alpindale/c4ai-command-r-plus-GPTQ": ["GPTQ", "Chat"],
    "meta-llama/Meta-Llama-3-70B-Instruct": [70.0, "Chat", "meta-llama/Meta-Llama-3-70B"],
    "MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ": ["AWQ", "Chat", "mistral-community/Mixtral-8x22B-v0.1-AWQ"],
    "MaziyarPanahi/zephyr-orpo-141b-A35b-v0.1-AWQ": ["AWQ", "Chat", "mistral-community/Mixtral-8x22B-v0.1-AWQ"],
    "Qwen/Qwen1.5-72B-Chat": [72.0, "Chat", "Qwen/Qwen1.5-72B"],
    "qwen/qwen-110b-chat": [110.0, "Chat", None],
    "gpt-3.5-turbo-1106": ["Proprietary", "Proprietary"],
    "gpt-3.5-turbo-0125": ["Proprietary", "Proprietary"],
    "gpt-4-1106-preview": ["Proprietary", "Proprietary"],
    "gpt-4-0125-preview": ["Proprietary", "Proprietary"],
    "gpt-4-turbo-2024-04-09": ["Proprietary", "Proprietary"],
    "gpt-4o-2024-05-13": ["Proprietary", "Proprietary"],
    "mistral-medium": ["Proprietary", "Proprietary"],
    "mistral-large": ["Proprietary", "Proprietary"],
    "gemini-1.0-pro": ["Proprietary", "Proprietary"],
    "gemini-pro-1.5": ["Proprietary", "Proprietary"],
    "google/gemini-flash-1.5": ["Proprietary", "Proprietary"],
    "claude-3-haiku-20240307": ["Proprietary", "Proprietary"],
    "claude-3-sonnet-20240229": ["Proprietary", "Proprietary"],
    "claude-3-opus-20240229": ["Proprietary", "Proprietary"],
}


MODEL_SHORT_TO_LONG = {model.split("/")[-1]: model for model in ORDERED_MODELS}


def get_model_type(model_name: str) -> str:
    for _, model_list in MODELS["pretrained"].items():
        if model_name in model_list:
            return "base"

    for _, model_list in MODELS["instruction_tuned"].items():
        if model_name in model_list:
            return "instruct"

    if model_name in API_MODELS:
        return "api"

    raise ValueError(f"Model {model_name} not found in model_list.py")
    return None


def get_open_model_list() -> list:
    all_models = []
    for _, model_list in MODELS["pretrained"].items():
        all_models.extend(model_list)

    for _, model_list in MODELS["instruction_tuned"].items():
        all_models.extend(model_list)

    return all_models


def get_all_model_list() -> list:
    all_models = []
    for _, model_list in MODELS["pretrained"].items():
        all_models.extend(model_list)

    for _, model_list in MODELS["instruction_tuned"].items():
        all_models.extend(model_list)

    all_models.extend(API_MODELS)

    return all_models


def get_pretrained_models() -> list:
    all_models = []
    for _, model_list in MODELS["pretrained"].items():
        all_models.extend(model_list)
    return all_models


def get_instruct_models() -> list:
    all_models = []
    for _, model_list in MODELS["instruction_tuned"].items():
        all_models.extend(model_list)
    return all_models


def get_model_params(model_name: str) -> int:
    for size_range, model_list in MODELS["pretrained"].items():
        if model_name in model_list:
            return int(size_range.split("B")[0].replace("<=", ""))

    for size_range, model_list in MODELS["instruction_tuned"].items():
        if model_name in model_list:
            return int(size_range.split("B")[0].replace("<=", ""))

    raise ValueError(f"Model {model_name} not found in model_list.py")


def get_model_num_gpus(model_name: str) -> int:
    model_params = get_model_params(model_name)
    num_gpus = {
        4: 1,
        7: 1,
        14: 2,
        50: 4,
        75: 8,
        175: 4,
    }[model_params]
    return num_gpus


def get_not_trained_models() -> list:
    all_models = get_all_model_list()
    trained_models = bgb_trained_models
    not_trained_models = [model for model in all_models if model not in trained_models]
    return not_trained_models


def is_trained_model(model_name: str) -> bool:
    return model_name in bgb_trained_models


if __name__ == "__main__":
    assert get_model_type("microsoft/phi-1"), "base"
    assert get_model_params("microsoft/phi-2"), 4

    models = get_all_model_list()

    model_list_str = ""
    for model in models:
        model_list_str += f'"{model}"\n'
    print(model_list_str)

    print(f"{len(models)} models found in src/model_list.py")

    print(get_not_trained_models())
