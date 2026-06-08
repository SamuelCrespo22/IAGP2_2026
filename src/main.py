import os
import csv
from dotenv import load_dotenv
from generator import LCMGenerator
from evaluator import ImageEvaluator
import utils

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS_DIR = os.path.join(BASE_DIR, "data")

NUM_ITERATIONS = 5
VARIANTS_PER_ITER = 5
NUM_RUNS = 15

EXPERIMENTS = [
    {
        "name": "Experiment 3: Gemini",
        "output_dir_name": "outputs_15_runs_combined_score_3",
        "prompts": {
            "9338.png": "fantasy painting of a mouse with intricate rainbow scales and horns within a vibrant, glowing abstract flame background"
        }
    }
]

client = utils.init_groq_client()


def select_best_candidate(candidates):
    return max(candidates, key=lambda item: item.get("Combined_Score", 0.0))


def rebuild_global_outputs(outputs_path, generator, num_runs=15):
    print("\nRebuilding global rankings using all existing image folders...")

    all_global_candidates = []

    for img_folder in sorted(os.listdir(outputs_path)):
        img_base_dir = os.path.join(outputs_path, img_folder)

        if not os.path.isdir(img_base_dir):
            continue

        if img_folder in ["global_winners"]:
            continue

        filename = img_folder + ".png"

        for run_id in range(1, num_runs + 1):
            csv_path = os.path.join(img_base_dir, f"run_{run_id}", "metrics_log.csv")

            if not os.path.exists(csv_path):
                continue

            df = utils.safe_read_csv(csv_path)
            if df is None:
                continue

            for _, row in df.iterrows():
                all_global_candidates.append({
                    "Image_Name": filename,
                    "Run_ID": f"run_{run_id}",
                    "Iteration": row["Iteration"],
                    "Prompt": row["Prompt"],
                    "Metrics": {
                        "CLIP_Sim": float(row["CLIP_Sim"]),
                        "LPIPS": float(row["LPIPS"]),
                        "RMSE": float(row["RMSE"])
                    },
                    "Image": None
                })

        utils.extract_top_3_for_report(filename, img_base_dir, generator, num_runs)

    global_csv_path = os.path.join(outputs_path, "global_absolute_ranking.csv")
    winners_dir = os.path.join(outputs_path, "global_winners")
    os.makedirs(winners_dir, exist_ok=True)

    image_names = sorted(set(item["Image_Name"] for item in all_global_candidates))

    with open(global_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image_Name",
            "Global_Rank",
            "Run_ID",
            "Iteration",
            "Prompt",
            "CLIP_Sim",
            "LPIPS",
            "RMSE",
            "Global_Combined_Score"
        ])

        for img_target in image_names:
            img_pool = [
                item for item in all_global_candidates
                if item["Image_Name"] == img_target
            ]

            if not img_pool:
                continue

            utils.compute_combined_scores(img_pool)
            img_pool.sort(
                key=lambda item: item.get("Combined_Score", 0.0),
                reverse=True
            )

            for rank, item in enumerate(img_pool, start=1):
                writer.writerow([
                    item["Image_Name"],
                    rank,
                    item["Run_ID"],
                    item["Iteration"],
                    item["Prompt"],
                    item["Metrics"]["CLIP_Sim"],
                    item["Metrics"]["LPIPS"],
                    item["Metrics"]["RMSE"],
                    f"{item.get('Combined_Score', 0.0):.6f}"
                ])

            best = img_pool[0]
            clean_name = img_target.split(".")[0]

            try:
                best_img = generator.generate(
                    prompt=best["Prompt"],
                    target_filename=img_target
                )
                utils.save_image(
                    best_img,
                    os.path.join(winners_dir, f"{clean_name}_best_match.png")
                )
            except Exception as e:
                print(f"[WARN] Could not render global winner for {img_target}: {e}")

    image_prompts = {img_name: "" for img_name in image_names}
    utils.calculate_global_statistics(image_prompts, outputs_path)

    print("[SUCCESS] Global ranking, global winners, top3 files, and aggregate stats rebuilt.")


def run_selected_prompts(exp, generator, evaluator):
    outputs_path = os.path.join(BASE_DIR, exp["output_dir_name"])
    os.makedirs(outputs_path, exist_ok=True)

    txt_path = os.path.join(outputs_path, "prompts_used_partial_run.txt")
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"=== PARTIAL RUN PROMPTS FOR: {exp['name']} ===\n\n")
        for img_key, prompt_text in exp["prompts"].items():
            txt_file.write(f"{img_key}:\n{prompt_text}\n\n")

    for filename, initial_prompt in exp["prompts"].items():
        print(f"\n{'=' * 50}")
        print(f"Running selected target: {filename}")
        print(f"{'=' * 50}")

        target_path = os.path.join(TARGETS_DIR, filename)
        target_img = utils.load_image(target_path)

        if target_img is None:
            print(f"[WARN] Target image not found: {target_path}")
            continue

        img_base_dir = os.path.join(outputs_path, filename.split(".")[0])
        os.makedirs(img_base_dir, exist_ok=True)

        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n--- {exp['output_dir_name']} | Run {run_id}/{NUM_RUNS} for {filename} ---")

            run_output_dir = os.path.join(img_base_dir, f"run_{run_id}")
            os.makedirs(run_output_dir, exist_ok=True)

            best_prompt = initial_prompt

            baseline = utils.generate_and_evaluate(
                best_prompt,
                generator,
                evaluator,
                target_img,
                filename
            )

            if baseline is None:
                continue

            base_img = baseline["Image"]
            best_metrics = baseline["Metrics"]

            utils.save_image(base_img, os.path.join(run_output_dir, "baseline.png"))

            all_run_candidates = []

            all_run_candidates.append({
                "Iteration": "Baseline",
                "Prompt": best_prompt,
                "Metrics": best_metrics,
                "Image": base_img
            })

            for iteration in range(NUM_ITERATIONS):
                candidate_prompts = utils.call_llm_for_prompts(
                    client,
                    best_prompt,
                    best_metrics["CLIP_Sim"],
                    best_metrics["RMSE"],
                    VARIANTS_PER_ITER
                )

                iteration_candidates = []

                for prompt in candidate_prompts:
                    cand = utils.generate_and_evaluate(
                        prompt,
                        generator,
                        evaluator,
                        target_img,
                        filename
                    )

                    if cand is None:
                        continue

                    cand_entry = {
                        "Iteration": iteration + 1,
                        "Prompt": prompt,
                        "Metrics": cand["Metrics"],
                        "Image": cand["Image"]
                    }

                    iteration_candidates.append(cand_entry)
                    all_run_candidates.append(cand_entry)

                utils.compute_combined_scores(all_run_candidates)

                current_pool = iteration_candidates + [{
                    "Prompt": best_prompt,
                    "Metrics": best_metrics,
                    "Image": base_img,
                    "Combined_Score": next(
                        (
                            x.get("Combined_Score", 0.0)
                            for x in all_run_candidates
                            if x["Prompt"] == best_prompt
                        ),
                        0.0
                    )
                }]

                current_best = select_best_candidate(current_pool)

                if current_best["Prompt"] != best_prompt:
                    best_prompt = current_best["Prompt"]
                    best_metrics = current_best["Metrics"]
                    base_img = current_best["Image"]

                    utils.save_image(
                        current_best["Image"],
                        os.path.join(run_output_dir, "current_best.png")
                    )

            utils.compute_combined_scores(all_run_candidates)

            csv_path = os.path.join(run_output_dir, "metrics_log.csv")
            with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Iteration",
                    "Prompt",
                    "CLIP_Sim",
                    "LPIPS",
                    "RMSE",
                    "Combined_Score"
                ])

                for item in all_run_candidates:
                    writer.writerow([
                        item["Iteration"],
                        item["Prompt"],
                        item["Metrics"]["CLIP_Sim"],
                        item["Metrics"]["LPIPS"],
                        item["Metrics"]["RMSE"],
                        f"{item.get('Combined_Score', 0.0):.6f}"
                    ])

        utils.extract_top_3_for_report(filename, img_base_dir, generator, NUM_RUNS)

    rebuild_global_outputs(outputs_path, generator, NUM_RUNS)


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] Groq API key not found! Confirm that the .env file exists.")
        return

    print("Loading LCM and evaluation models globally...")
    generator, evaluator = utils.init_generator_evaluator(
        LCMGenerator,
        ImageEvaluator
    )

    if generator is None or evaluator is None:
        print("[ERROR] Generator or evaluator initialization failed. Aborting.")
        return

    for exp in EXPERIMENTS:
        print(f"\n\n############# STARTING {exp['name'].upper()} #############")
        run_selected_prompts(exp, generator, evaluator)
        print(f"[SUCCESS] Completed partial run for {exp['name']}.")

    print("\nAll selected prompts completed and global files rebuilt.")


if __name__ == "__main__":
    main()