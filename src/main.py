import os
import csv
from dotenv import load_dotenv
from generator import LCMGenerator
from evaluator import ImageEvaluator
import utils

load_dotenv()

# ---------------------------------------------------------
# Initial Configuration
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs_15_runs_combined_score")
NUM_ITERATIONS = 5      # Optimization: how many times to try to improve the prompt
VARIANTS_PER_ITER = 5   # Optimization: how many new prompts per iteration
NUM_RUNS = 15           # Requirement: repetitions of the cycle per image (at least 5)

INITIAL_PROMPTS = {
    "1159_7.png": "there is a small hedgehog sitting on top of a block of cake",
    "7836.png": "astronaut standing on the surface of a planet with a bright light shining in the background",
    "1159_25.png": "there is a glass of orange juice with a slice of orange on the side",
    "1159_29.png": "arafed palm tree on a rock in the ocean at sunset",
    "1159_3.png": "anime character with a sword and fire in his hand",
    "9338.png": "painting of a mouse with a colorful tail and tail"
}

client = utils.init_groq_client()

# ---------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------
def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] Groq API key not found! Confirm that the .env file exists.")
        return

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    print("Loading LCM and evaluation models...")
    generator, evaluator = utils.init_generator_evaluator(LCMGenerator, ImageEvaluator)
    if generator is None or evaluator is None:
        print("[ERROR] Generator or evaluator initialization failed. Aborting.")
        return
    
    for filename, initial_prompt in INITIAL_PROMPTS.items():
        print(f"\n{'='*50}\n Processing image: {filename} \n{'='*50}")
        
        target_path = os.path.join(TARGETS_DIR, filename)
        target_img = utils.load_image(target_path)
        if target_img is None:
            continue
        
        img_base_dir = os.path.join(OUTPUTS_DIR, filename.split('.')[0])
        os.makedirs(img_base_dir, exist_ok=True)
        
        # Do N repetitions (optimizer seeds)
        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n--- STARTING RUN {run_id}/{NUM_RUNS} for {filename} ---")
            
            run_output_dir = os.path.join(img_base_dir, f"run_{run_id}")
            os.makedirs(run_output_dir, exist_ok=True)
            
            best_prompt = initial_prompt
            
            # Baseline for this run
            baseline = utils.generate_and_evaluate(best_prompt, generator, evaluator, target_img, filename)
            if baseline is None:
                print(f"[ERROR] Baseline failed for {filename} in run {run_id}; skipping run.")
                continue

            base_img = baseline['Image']
            best_metrics = baseline['Metrics']

            if not utils.save_image(base_img, os.path.join(run_output_dir, "baseline.png")):
                print(f"[ERROR] Failed to save baseline image for {filename} run {run_id}")
            
            csv_path = os.path.join(run_output_dir, "metrics_log.csv")
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Prompt", "CLIP_Sim", "LPIPS", "RMSE", "Combined_Score"])
                writer.writerow(["Baseline", best_prompt, best_metrics['CLIP_Sim'], best_metrics['LPIPS'], best_metrics['RMSE'], ""])
            
            best_combined = None
            
            # Iteration loop (LLM fine-tuning)
            for iteration in range(NUM_ITERATIONS):
                print(f"  Iteration {iteration+1}/{NUM_ITERATIONS}...")
                
                candidate_prompts = utils.call_llm_for_prompts(client, best_prompt, best_metrics['CLIP_Sim'], best_metrics['RMSE'], VARIANTS_PER_ITER)
                iteration_candidates = []

                for prompt in candidate_prompts:
                    cand = utils.generate_and_evaluate(prompt, generator, evaluator, target_img, filename)
                    if cand is None:
                        continue
                    iteration_candidates.append({
                        'Prompt': prompt,
                        'Metrics': cand['Metrics'],
                        'Image': cand['Image']
                    })

                scoring_candidates = iteration_candidates + [{
                    'Prompt': best_prompt,
                    'Metrics': best_metrics,
                    'Image': base_img
                }]
                utils.compute_combined_scores(scoring_candidates)

                for candidate in iteration_candidates:
                    utils.append_metrics_csv(csv_path, [
                        iteration+1,
                        candidate['Prompt'],
                        candidate['Metrics']['CLIP_Sim'],
                        candidate['Metrics']['LPIPS'],
                        candidate['Metrics']['RMSE'],
                        f"{candidate['Combined_Score']:.6f}"
                    ])

                current_best = max(scoring_candidates, key=lambda item: item['Combined_Score'])

                if best_combined is None or current_best['Combined_Score'] > best_combined:
                    best_combined = current_best['Combined_Score']
                    best_prompt = current_best['Prompt']
                    best_metrics = current_best['Metrics']
                    base_img = current_best['Image']

                    if current_best['Image'] is not None:
                        utils.safe_save_image(current_best['Image'], os.path.join(run_output_dir, "current_best.png"))

                print(f"    Best candidate this iteration: {best_prompt} (combined={best_combined:.6f})")

            print(f"-> Winner (Run {run_id}): {best_prompt} (CLIP: {best_metrics['CLIP_Sim']:.4f}, combined: {best_combined:.6f})")

        utils.extract_top_3_for_report(filename, img_base_dir, generator, NUM_RUNS)
    
    utils.calculate_global_statistics(INITIAL_PROMPTS, OUTPUTS_DIR)

if __name__ == "__main__":
    main()