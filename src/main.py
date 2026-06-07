import os
import csv
import shutil
from dotenv import load_dotenv
from generator import LCMGenerator
from evaluator import ImageEvaluator
import utils

load_dotenv()

# ---------------------------------------------------------
# Structural Hyperparameters
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS_DIR = os.path.join(BASE_DIR, "data")

NUM_ITERATIONS = 5      # Optimization: how many times to try to improve the prompt
VARIANTS_PER_ITER = 5   # Optimization: how many new prompts per iteration
NUM_RUNS = 15           # Requirement: repetitions of the cycle per image

# ---------------------------------------------------------
# Define the Three Experimental Configurations
# ---------------------------------------------------------
EXPERIMENTS = [
    {
        "name": "Experiment 4",
        "output_dir_name": "outputs_15_runs_combined_score_4",
        "prompts": {
            "1159_7.png": "photorealistic hedgehog with golden spiky fur curled on top of a geometric stacked cube, soft beige studio lighting, surreal 3D render",
            "7836.png": "lone astronaut in white spacesuit on alien planet surface, massive ringed planet looming overhead, blue nebula and stars, cinematic sci-fi scene",
            "1159_25.png": "glass of fresh orange juice with orange slice garnish and ice, sliced oranges scattered around, warm dramatic food photography lighting",
            "1159_29.png": "lone palm tree standing in shallow turquoise ocean waves, golden sunset on horizon, distant mountains, serene tropical digital art",
            "1159_3.png": "blonde male spiky-haired anime warrior in silver armor, glowing golden energy whip, teal and orange fire auras, dark dramatic background, semi-realistic fantasy art",
            "9338.png": "cute hamster-dragon hybrid with colorful dragon scales and small horns, curled tail, vibrant rainbow flame background, impressionistic digital painting"
        }
    }
]

client = utils.init_groq_client()

def select_best_candidate(candidates):
    return max(candidates, key=lambda item: item.get('Combined_Score', 0.0))

# ---------------------------------------------------------
# Execution Loop
# ---------------------------------------------------------
def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] Groq API key not found! Confirm that the .env file exists.")
        return

    print("Loading LCM and evaluation models globally...")
    generator, evaluator = utils.init_generator_evaluator(LCMGenerator, ImageEvaluator)
    if generator is None or evaluator is None:
        print("[ERROR] Generator or evaluator initialization failed. Aborting.")
        return

    # Cycle through each experimental block sequentially
    for exp in EXPERIMENTS:
        print(f"\n\n############# STARTING {exp['name'].upper()} #############")
        
        outputs_path = os.path.join(BASE_DIR, exp['output_dir_name'])
        os.makedirs(outputs_path, exist_ok=True)
        
        # Save the prompt configurations to a text file inside this specific directory
        txt_path = os.path.join(outputs_path, "prompts_used.txt")
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(f"=== INITIAL PROMPTS FOR: {exp['name']} ===\n\n")
            for img_key, prompt_text in exp['prompts'].items():
                txt_file.write(f"{img_key}:\n{prompt_text}\n\n")
        
        all_global_candidates = []
        initial_prompts_dict = exp['prompts']

        for filename, initial_prompt in initial_prompts_dict.items():
            print(f"\n{'='*50}\n Target Asset: {filename} \n{'='*50}")
            
            target_path = os.path.join(TARGETS_DIR, filename)
            target_img = utils.load_image(target_path)
            if target_img is None:
                continue
            
            img_base_dir = os.path.join(outputs_path, filename.split('.')[0])
            os.makedirs(img_base_dir, exist_ok=True)
            
            for run_id in range(1, NUM_RUNS + 1):
                print(f"\n--- {exp['output_dir_name']} | Run {run_id}/{NUM_RUNS} for {filename} ---")
                
                run_output_dir = os.path.join(img_base_dir, f"run_{run_id}")
                os.makedirs(run_output_dir, exist_ok=True)
                
                best_prompt = initial_prompt
                
                baseline = utils.generate_and_evaluate(best_prompt, generator, evaluator, target_img, filename)
                if baseline is None:
                    continue

                base_img = baseline['Image']
                best_metrics = baseline['Metrics']

                utils.save_image(base_img, os.path.join(run_output_dir, "baseline.png"))
                csv_path = os.path.join(run_output_dir, "metrics_log.csv")
                
                all_run_candidates = []
                baseline_entry = {
                    'Iteration': 'Baseline',
                    'Prompt': best_prompt,
                    'Metrics': best_metrics,
                    'Image': base_img
                }
                all_run_candidates.append(baseline_entry)
                
                for iteration in range(NUM_ITERATIONS):
                    candidate_prompts = utils.call_llm_for_prompts(client, best_prompt, best_metrics['CLIP_Sim'], best_metrics['RMSE'], VARIANTS_PER_ITER)
                    iteration_candidates = []

                    for prompt in candidate_prompts:
                        cand = utils.generate_and_evaluate(prompt, generator, evaluator, target_img, filename)
                        if cand is None:
                            continue
                        
                        cand_entry = {
                            'Iteration': iteration + 1,
                            'Prompt': prompt,
                            'Metrics': cand['Metrics'],
                            'Image': cand['Image']
                        }
                        iteration_candidates.append(cand_entry)
                        all_run_candidates.append(cand_entry)

                    utils.compute_combined_scores(all_run_candidates)

                    current_pool = iteration_candidates + [{
                        'Prompt': best_prompt,
                        'Metrics': best_metrics,
                        'Image': base_img,
                        'Combined_Score': next((x.get('Combined_Score', 0.0) for x in all_run_candidates if x['Prompt'] == best_prompt), 0.0)
                    }]
                    
                    current_best = select_best_candidate(current_pool)

                    if current_best['Prompt'] != best_prompt:
                        best_prompt = current_best['Prompt']
                        best_metrics = current_best['Metrics']
                        base_img = current_best['Image']
                        utils.save_image(current_best['Image'], os.path.join(run_output_dir, "current_best.png"))

                # Finalize run calibrations
                utils.compute_combined_scores(all_run_candidates)
                with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Iteration", "Prompt", "CLIP_Sim", "LPIPS", "RMSE", "Combined_Score"])
                    for item in all_run_candidates:
                        writer.writerow([
                            item['Iteration'], item['Prompt'], item['Metrics']['CLIP_Sim'],
                            item['Metrics']['LPIPS'], item['Metrics']['RMSE'], f"{item.get('Combined_Score', 0.0):.6f}"
                        ])

                for item in all_run_candidates:
                    all_global_candidates.append({
                        'Image_Name': filename,
                        'Run_ID': f"run_{run_id}",
                        'Iteration': item['Iteration'],
                        'Prompt': item['Prompt'],
                        'Metrics': item['Metrics'],
                        'Image': item['Image']
                    })

            utils.extract_top_3_for_report(filename, img_base_dir, generator, NUM_RUNS)
        
        # Post-process current experiment (Global Leaderboard per directory)
        global_csv_path = os.path.join(outputs_path, "global_absolute_ranking.csv")
        winners_dir = os.path.join(outputs_path, "global_winners")
        os.makedirs(winners_dir, exist_ok=True)

        with open(global_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Image_Name", "Global_Rank", "Run_ID", "Iteration", "Prompt", "CLIP_Sim", "LPIPS", "RMSE", "Global_Combined_Score"])

            for img_target in initial_prompts_dict.keys():
                img_pool = [x for x in all_global_candidates if x['Image_Name'] == img_target]
                if not img_pool:
                    continue

                utils.compute_combined_scores(img_pool)
                img_pool.sort(key=lambda item: item.get('Combined_Score', 0.0), reverse=True)

                for rank, item in enumerate(img_pool, start=1):
                    writer.writerow([
                        item['Image_Name'], rank, item['Run_ID'], item['Iteration'],
                        item['Prompt'], item['Metrics']['CLIP_Sim'], item['Metrics']['LPIPS'],
                        item['Metrics']['RMSE'], f"{item.get('Combined_Score', 0.0):.6f}"
                    ])

                absolute_champion = img_pool[0]
                if absolute_champion['Image'] is not None:
                    clean_name = img_target.split('.')[0]
                    utils.save_image(absolute_champion['Image'], os.path.join(winners_dir, f"{clean_name}_best_match.png"))

        # Generate your structural global statistics summary for this experiment
        utils.calculate_global_statistics(initial_prompts_dict, outputs_path)
        print(f"[SUCCESS] Completed {exp['name']}. Data saved to folder: {exp['output_dir_name']}\n")

    print("\nAll experiments successfully verified and completed.")

if __name__ == "__main__":
    main()