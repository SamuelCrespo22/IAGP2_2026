import os
import time
import csv
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# External service initialization
# ---------------------------------------------------------------------------

def init_groq_client():
    try:
        from groq import Groq
        return Groq()
    except Exception as e:
        print(f"[ERROR] Groq client initialization failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Device and randomness helpers
# ---------------------------------------------------------------------------

def get_compute_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_torch_deterministic():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def seed_torch(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def create_torch_generator(seed: int, device=None):
    if device is None:
        device = get_compute_device()
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return generator


def get_seed_from_filename(filename):
    base = os.path.basename(filename)
    seed_str = base.split('_')[0].split('.')[0]
    try:
        return int(seed_str)
    except ValueError:
        print(f"[ERROR] Unable to parse seed from filename {filename}; defaulting to 0")
        return 0


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def init_generator_evaluator(generator_cls=None, evaluator_cls=None):
    generator = None
    evaluator = None

    if generator_cls is not None:
        try:
            generator = generator_cls()
        except Exception as e:
            print(f"[ERROR] Failed to initialize generator: {e}")
            generator = None

    if evaluator_cls is not None:
        try:
            evaluator = evaluator_cls()
        except Exception as e:
            print(f"[ERROR] Failed to initialize evaluator: {e}")
            evaluator = None

    return generator, evaluator


# ---------------------------------------------------------------------------
# File and image helpers
# ---------------------------------------------------------------------------

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV {path}: {e}")
        return None


def save_dataframe(df, path):
    try:
        df.to_csv(path, index=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save DataFrame to {path}: {e}")
        return False


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to open image {path}: {e}")
        return None


def save_image(img, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save image {path}: {e}")
        return False


def append_metrics_csv(csv_path, row):
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to append to CSV {csv_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Generation and evaluation helpers
# ---------------------------------------------------------------------------

def generate_and_evaluate(prompt, generator, evaluator, target_img, target_filename):
    try:
        img = generator.generate(prompt=prompt, target_filename=target_filename)
        metrics = evaluator.evaluate(target_img, img)
        return {"Image": img, "Metrics": metrics}
    except Exception as e:
        print(f"[WARN] Generation/evaluation failed for prompt '{prompt}': {e}")
        return None


# ---------------------------------------------------------------------------
# Metrics scoring helpers
# ---------------------------------------------------------------------------

def normalize_values(values):
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.5 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def compute_combined_scores(entries):
    clip_scores = [item['Metrics']['CLIP_Sim'] for item in entries]
    lpips_scores = [item['Metrics']['LPIPS'] for item in entries]
    rmse_scores = [item['Metrics']['RMSE'] for item in entries]

    clip_norm = normalize_values(clip_scores)
    lpips_norm = normalize_values(lpips_scores)
    rmse_norm = normalize_values(rmse_scores)

    for idx, item in enumerate(entries):
        item['Combined_Score'] = (
            0.50 * clip_norm[idx]
            + 0.25 * (1.0 - lpips_norm[idx])
            + 0.25 * (1.0 - rmse_norm[idx])
        )

    return entries


def compute_combined_scores_df(df):
    df = df.copy()
    df['CLIP_norm'] = normalize_values(df['CLIP_Sim'].tolist())
    df['LPIPS_norm'] = normalize_values(df['LPIPS'].tolist())
    df['RMSE_norm'] = normalize_values(df['RMSE'].tolist())
    df['Combined_Score'] = (
        0.50 * df['CLIP_norm']
        + 0.25 * (1.0 - df['LPIPS_norm'])
        + 0.25 * (1.0 - df['RMSE_norm'])
    )
    return df


# ---------------------------------------------------------------------------
# LLM prompt generation helpers
# ---------------------------------------------------------------------------

def call_llm_for_prompts(client, current_prompt, clip_score, rmse_score, variants=5):
    time.sleep(2)

    system_prompt = (
        "You are an expert prompt engineer for Stable Diffusion models. "
        "Your goal is to perfectly reconstruct a target image by tweaking the prompt. "
        "Respond ONLY with EXACTLY 5 new prompt variations separated by the pipe character '|'. "
        "Do not include numbering, explanations, quotes, or conversational text. "
        "Just the 5 phrases separated by '|'."
    )

    user_prompt = (
        f"The current best prompt is: '{current_prompt}'.\n"
        f"Its CLIP similarity score is {clip_score:.4f} (higher is better, 1.0 is perfect).\n"
        f"Its pixel RMSE error is {rmse_score:.4f} (lower is better, 0.0 is perfect).\n\n"
        f"Generate 5 variations of this prompt to try and improve these scores. "
        f"Try altering descriptive words, fixing hallucinations, adding style keywords "
        f"(e.g., highly detailed, 4k, masterpiece, digital painting), "
        f"or slightly changing the framing. "
        f"Format strictly as: variation 1 | variation 2 | variation 3 | variation 4 | variation 5"
    )

    if client is None:
        return [
            current_prompt + " hd",
            current_prompt + " 4k",
            current_prompt + " 8k",
            current_prompt + " masterpiece",
            current_prompt + " ultra detailed"
        ]

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9
        )

        raw_text = response.choices[0].message.content.strip()
        variations = [v.strip() for v in raw_text.split("|") if v.strip()]

        if len(variations) < variants:
            variations = [
                current_prompt + ", highly detailed",
                current_prompt + ", high quality, sharp",
                current_prompt + ", best lighting",
                current_prompt + ", masterpiece",
                current_prompt + ", ultra realistic"
            ]

        return variations[:variants]

    except Exception as e:
        print(f"  [Groq API error]: {e}")
        return [
            current_prompt + " hd",
            current_prompt + " 4k",
            current_prompt + " 8k",
            current_prompt + " masterpiece",
            current_prompt + " ultra detailed"
        ]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def calculate_global_statistics(initial_prompts, outputs_dir):
    print("\nCalculating global average and standard deviation for all Test Set...")
    all_top3_data = []

    for filename in initial_prompts.keys():
        img_name = filename.split('.')[0]
        csv_path = os.path.join(outputs_dir, img_name, "top3_final", "top3_metrics.csv")

        if os.path.exists(csv_path):
            df = safe_read_csv(csv_path)
            if df is not None:
                all_top3_data.append(df)

    if not all_top3_data:
        print("[WARN] No top3 data found to compute global statistics.")
        return

    global_df = pd.concat(all_top3_data, ignore_index=True)

    try:
        stats = global_df[['CLIP_Sim', 'LPIPS', 'RMSE']].agg(['mean', 'std']).round(4)
        stats_path = os.path.join(outputs_dir, "global_aggregate_stats.csv")
        stats.to_csv(stats_path)
        print(f"-> '{stats_path}' file saved with success!")
    except Exception as e:
        print(f"[WARN] Failed to compute/save global statistics: {e}")


def extract_top_3_for_report(filename, img_base_dir, generator, num_runs=15, top_n=3):
    print(f"\nCompiling final Top-{top_n} for {filename}...")

    all_runs_data = []
    for run_id in range(1, num_runs + 1):
        csv_path = os.path.join(img_base_dir, f"run_{run_id}", "metrics_log.csv")
        if os.path.exists(csv_path):
            df = safe_read_csv(csv_path)
            if df is not None:
                df['Run'] = f"run_{run_id}"
                all_runs_data.append(df)

    if not all_runs_data:
        print(f"[WARN] No run CSV data found for {filename}. Skipping Top-{top_n} extraction.")
        return False

    combined_df = pd.concat(all_runs_data, ignore_index=True)
    combined_df = compute_combined_scores_df(combined_df)

    sorted_df = combined_df.sort_values(by="Combined_Score", ascending=False)
    top_df = sorted_df.drop_duplicates(subset=['Prompt'], keep='first').head(top_n)

    top3_dir = os.path.join(img_base_dir, "top3_final")
    os.makedirs(top3_dir, exist_ok=True)

    if not save_dataframe_safe(top_df, os.path.join(top3_dir, "top3_metrics.csv")):
        print(f"[WARN] Failed to save Top-{top_n} CSV for {filename}.")

    if generator is None:
        print(f"[WARN] Generator not available for Top-{top_n} rendering of {filename}.")
        return False

    print("  Rendering the official images...")
    for idx, row in enumerate(top_df.itertuples(), start=1):
        prompt = row.Prompt
        score = getattr(row, 'Combined_Score', None)
        try:
            img = generator.generate(prompt=prompt, target_filename=filename)
            safe_save_image(img, os.path.join(top3_dir, f"rank_{idx}_cand.png"))
            print(f"    Saved Rank {idx} (combined={score:.6f}): '{prompt}'")
        except Exception as e:
            print(f"[WARN] Failed to render/save Top-{top_n} image rank {idx} for {filename}: {e}")

    return True
