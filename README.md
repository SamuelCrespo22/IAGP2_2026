# IAGTP2: Image-to-Prompt Inversion

This repository contains the implementation of Project 2 for the Generative AI course (2025/2026).

## Overview

The objective of this project is to reconstruct target images by searching for prompts capable of generating visually similar outputs under a fixed diffusion pipeline.

The proposed methodology combines:

1. Semantic initialization using Gemini-generated image descriptions.
2. Iterative prompt refinement using a Large Language Model (LLM).
3. Image evaluation using CLIP Similarity, LPIPS, and Pixel RMSE.
4. Candidate ranking through a Combined Score.

The main pipeline is implemented in `src/main.py`, image generation is handled by `src/generator.py`, and image evaluation is implemented in `src/evaluator.py`.

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure the API Key

Create a `.env` file and provide your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Prepare the Target Images

Place the target images inside the appropriate data directory used by the project.

---

## Running the Pipeline

Execute the main pipeline:

```bash
python -m src.main
```

The pipeline will:

1. Load the target image.
2. Initialize the search using a Gemini-generated prompt.
3. Perform iterative LLM-based prompt refinement.
4. Evaluate generated images using CLIP, LPIPS, and RMSE.
5. Repeat the optimization process across multiple runs.
6. Select the final Top-3 prompts for each target image.

---

## Output Structure

Results are saved inside the selected experiment directory (for example, `outputs_15_runs_combined_score_3/`).

Typical outputs include:

* `run_<i>/metrics_log.csv` — candidate metrics for each run.
* `top3_final/top3_metrics.csv` — metrics for the final Top-3 prompts.
* `global_absolute_ranking.csv` — global ranking of all candidates.
* `global_aggregate_stats.csv` — aggregated statistics across the dataset.

---

## Methodology Summary

### Stage 1 — Semantic Initialization

Several initialization strategies were explored, including:

* BLIP captions
* ChatGPT prompts
* Gemini descriptions
* Nano Banana prompts

Gemini produced the strongest starting prompts and was therefore selected for the final system.

For each target image, Gemini generates an initial textual description containing the main subjects, scene composition, visual attributes, and stylistic characteristics. These prompts serve as the starting point for the optimization process.

### Stage 2 — LLM-Based Refinement

Starting from the initial Gemini prompt, Llama-3.1 (8B) generates prompt variations that are rendered using the fixed diffusion pipeline and evaluated against the target image.

Each optimization run consists of:

* 5 refinement iterations
* 5 prompt candidates per iteration

The complete process is repeated 15 times for each target image to improve robustness against stochastic LLM behavior.

### Candidate Ranking

Generated candidates are evaluated using:

* CLIP Similarity
* LPIPS
* Pixel RMSE

These metrics are combined into a ranking score:

```text
Combined Score =
0.50 × CLIP_norm +
0.25 × (1 − LPIPS_norm) +
0.25 × (1 − RMSE_norm)
```

The Combined Score is used exclusively for candidate ranking and Top-3 selection.

---

## Deterministic Generation Setup

All image generations use a fixed configuration:

* Model: `SimianLuo/LCM_Dreamshaper_v7`
* Resolution: 768 × 768
* Inference Steps: 8
* Guidance Scale: 8.0
* LCM Origin Steps: 50

The generation seed is extracted from the target image filename and remains fixed throughout all experiments.

Under identical software and hardware conditions, the same prompt and target image will produce identical generated images.

---

## Main Files

* `src/main.py` — main optimization pipeline.
* `src/generator.py` — LCM image generation.
* `src/evaluator.py` — CLIP, LPIPS, and RMSE evaluation.
* `requirements.txt` — project dependencies.
* `.env` — Groq API configuration.

---

## Notes

* The generator configuration remains fixed throughout all experiments.
* The rendering process is deterministic for a given prompt and target image.
* Final performance is reported using CLIP Similarity, LPIPS, and RMSE.
* The Combined Score is used only for candidate ranking and Top-3 selection.
