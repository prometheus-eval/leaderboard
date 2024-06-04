LOGO = '<img src="https://raw.githubusercontent.com/prometheus-eval/leaderboard/main/logo.png">'

TITLE = """<h1 align="center" id="space-title">🤗 BiGGen-Bench Leaderboard 🏋️</h1>"""

BGB_LOGO = '<img src="https://raw.githubusercontent.com/prometheus-eval/leaderboard/main/logo.png" alt="Logo" style="width: 30%; display: block; margin: auto;">'
BGB_TITLE = """<h1 align="center">BiGGen-Bench Leaderboard</h1>"""



ABOUT = """
## 📝 About
The 🤗 LLM-Perf Leaderboard 🏋️ is a laderboard at the intersection of quality and performance.
Its aim is to benchmark the performance (latency, throughput, memory & energy) 
of Large Language Models (LLMs) with different hardwares, backends and optimizations 
using [Optimum-Benhcmark](https://github.com/huggingface/optimum-benchmark).

Anyone from the community can request a new base model or hardware/backend/optimization 
configuration for automated benchmarking:

- Model evaluation requests should be made in the 
[🤗 Open LLM Leaderboard 🏅](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) ;
we scrape the [list of canonical base models](https://github.com/huggingface/optimum-benchmark/blob/main/llm_perf/utils.py) from there.
- Hardware/Backend/Optimization configuration requests should be made in the 
[🤗 LLM-Perf Leaderboard 🏋️](https://huggingface.co/spaces/optimum/llm-perf-leaderboard) or 
[Optimum-Benhcmark](https://github.com/huggingface/optimum-benchmark) repository (where the code is hosted).

## ✍️ Details

- To avoid communication-dependent results, only one GPU is used.
- Score is the average evaluation score obtained from the [🤗 Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- LLMs are running on a singleton batch with a prompt size of 256 and generating a 64 tokens for at least 10 iterations and 10 seconds.
- Energy consumption is measured in kWh using CodeCarbon and taking into consideration the GPU, CPU, RAM and location of the machine.
- We measure three types of memory: Max Allocated Memory, Max Reserved Memory and Max Used Memory. The first two being reported by PyTorch and the last one being observed using PyNVML.

All of our benchmarks are ran by this single script
[benchmark_cuda_pytorch.py](https://github.com/huggingface/optimum-benchmark/blob/llm-perf/llm-perf/benchmark_cuda_pytorch.py)
using the power of [Optimum-Benhcmark](https://github.com/huggingface/optimum-benchmark) to garantee reproducibility and consistency.
"""


CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results."
CITATION_BUTTON = r"""@misc{llm-perf-leaderboard,
  author = {Ilyas Moutawwakil, Régis Pierrard},
  title = {LLM-Perf Leaderboard},
  year = {2023},
  publisher = {Hugging Face},
  howpublished = "\url{https://huggingface.co/spaces/optimum/llm-perf-leaderboard}",
@software{optimum-benchmark,
  author = {Ilyas Moutawwakil, Régis Pierrard},
  publisher = {Hugging Face},
  title = {Optimum-Benchmark: A framework for benchmarking the performance of Transformers models with different hardwares, backends and optimizations.},
}
"""
