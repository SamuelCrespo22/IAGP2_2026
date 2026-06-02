import os
import csv
import pandas as pd
import shutil
import time
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
from generator import LCMGenerator
from evaluator import ImageEvaluator

load_dotenv()

# ---------------------------------------------------------
# Initial Configuration
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
NUM_ITERATIONS = 5      # Optimization: how many times to try to improve the prompt
VARIANTS_PER_ITER = 3   # Optimization: how many new prompts per iteration
NUM_RUNS = 5            # Requirement: repetitions of the cycle per image (at least 5)

# Colab prompt dictionary
INITIAL_PROMPTS = {
    "1159_7.png": "there is a small hedgehog sitting on top of a block of cake"
    #"7836.png": "astronaut standing on the surface of a planet with a bright light shining in the background",
    #"1159_25.png": "there is a glass of orange juice with a slice of orange on the side",
    #"1159_29.png": "arafed palm tree on a rock in the ocean at sunset",
    #"1159_3.png": "anime character with a sword and fire in his hand",
    #"9338.png": "painting of a mouse with a colorful tail and tail"
}

client = Groq()

def calculate_global_statistics():
    print("\nCalculating global average and standard deviation for all Test Set...")
    all_top3_data = []
    
    for filename in INITIAL_PROMPTS.keys():
        img_name = filename.split('.')[0]
        csv_path = os.path.join(OUTPUTS_DIR, img_name, "top3_final", "top3_metrics.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_top3_data.append(df)
            
    if not all_top3_data:
        return
        
    global_df = pd.concat(all_top3_data, ignore_index=True)
    
    stats = global_df[['CLIP_Sim', 'LPIPS', 'RMSE']].agg(['mean', 'std']).round(4)
    
    stats.to_csv(os.path.join(OUTPUTS_DIR, "global_aggregate_stats.csv"))
    print("-> 'global_aggregate_stats.csv' file saved with success!")


# ---------------------------------------------------------
# Final step: extract final Top-3 for the report
# ---------------------------------------------------------
def extract_top_3_for_report(filename, img_base_dir):
    print(f"\nCompiling final Top-3 for {filename}...")
    
    # 1. Read all CSVs from all runs and combine them
    all_runs_data = []
    for run_id in range(1, NUM_RUNS + 1):
        csv_path = os.path.join(img_base_dir, f"run_{run_id}", "metrics_log.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Add a column to track source run
            df['Run'] = f"run_{run_id}" 
            all_runs_data.append(df)
            
    if not all_runs_data:
        return
        
    combined_df = pd.concat(all_runs_data, ignore_index=True)
    
    # 2. Sort by CLIP_Sim (highest to lowest) and remove exact prompt duplicates
    # Optional: you can sort first by RMSE (lowest to highest) if you prefer that metric
    sorted_df = combined_df.sort_values(by="CLIP_Sim", ascending=False)
    top_3_df = sorted_df.drop_duplicates(subset=['Prompt'], keep='first').head(3)
    
    # 3. Create official Top-3 folder
    top3_dir = os.path.join(img_base_dir, "top3_final")
    os.makedirs(top3_dir, exist_ok=True)
    
    # 4. Save cleaned CSV (your table for the report)
    top_3_df.to_csv(os.path.join(top3_dir, "top3_metrics.csv"), index=False)
    
    # 5. Regenerate the Top-3 images (safer and faster than searching the folder structure)
    generator = LCMGenerator() 
    print("  Rendering the 3 official images...")
    
    for idx, row in enumerate(top_3_df.itertuples(), start=1):
        prompt = row.Prompt
        score = row.CLIP_Sim
        
        # Render with the same sacred seed
        img = generator.generate(prompt=prompt, target_filename=filename)
        
        # Save with a clear name
        img.save(os.path.join(top3_dir, f"rank_{idx}_cand.png"))
        print(f"    Saved Rank {idx} (CLIP {score:.4f}): '{prompt}'")


# ---------------------------------------------------------
# LLM Integration (Groq / Llama-3)
# ---------------------------------------------------------
def call_llm_for_prompts(current_prompt, clip_score, rmse_score):
    time.sleep(2)

    system_prompt = (
        "You are an expert prompt engineer for Stable Diffusion models. "
        "Your goal is to perfectly reconstruct a target image by tweaking the prompt. "
        "Respond ONLY with EXACTLY 3 new prompt variations separated by the pipe character '|'. "
        "Do not include numbering, explanations, quotes, or conversational text. Just the 3 phrases separated by '|'."
    )
    
    user_prompt = (
        f"The current best prompt is: '{current_prompt}'.\n"
        f"Its CLIP similarity score is {clip_score:.4f} (higher is better, 1.0 is perfect).\n"
        f"Its pixel RMSE error is {rmse_score:.4f} (lower is better, 0.0 is perfect).\n\n"
        f"Generate 3 variations of this prompt to try and improve these scores. "
        f"Try altering descriptive words, fixing hallucinations, adding style keywords (e.g., highly detailed, 4k, masterpiece, digital painting), "
        f"or slightly changing the framing. "
        f"Format strictly as: variation 1 | variation 2 | variation 3"
    )

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7 
        )
        
        raw_text = response.choices[0].message.content.strip()
        variations = [v.strip() for v in raw_text.split('|') if v.strip()]
        
        if len(variations) < VARIANTS_PER_ITER:
            variations = [
                current_prompt + ", highly detailed", 
                current_prompt + ", high quality, sharp", 
                current_prompt + ", best lighting"
            ]
            
        return variations[:VARIANTS_PER_ITER]
        
    except Exception as e:
        print(f"  [Groq API error]: {e}")
        return [current_prompt + " hd", current_prompt + " 4k", current_prompt + " 8k"]

# ---------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------
def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("CRITICAL ERROR: Groq API key not found! Confirm that the .env file exists.")
        return

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    print("Loading LCM and evaluation models on RTX 3070...")
    generator = LCMGenerator()
    evaluator = ImageEvaluator()
    
    for filename, initial_prompt in INITIAL_PROMPTS.items():
        print(f"\n{'='*50}\n Processing image: {filename} \n{'='*50}")
        
        target_path = os.path.join(TARGETS_DIR, filename)
        target_img = Image.open(target_path).convert("RGB")
        
        img_base_dir = os.path.join(OUTPUTS_DIR, filename.split('.')[0])
        os.makedirs(img_base_dir, exist_ok=True)
        
        # Do N repetitions (optimizer seeds)
        for run_id in range(1, NUM_RUNS + 1):
            print(f"\n--- STARTING RUN {run_id}/{NUM_RUNS} for {filename} ---")
            
            run_output_dir = os.path.join(img_base_dir, f"run_{run_id}")
            os.makedirs(run_output_dir, exist_ok=True)
            
            best_prompt = initial_prompt
            
            # Baseline for this run
            base_img = generator.generate(prompt=best_prompt, target_filename=filename)
            base_metrics = evaluator.evaluate(target_img, base_img)
            best_score = base_metrics["CLIP_Sim"]
            
            base_img.save(os.path.join(run_output_dir, "baseline.png"))
            
            csv_path = os.path.join(run_output_dir, "metrics_log.csv")
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Iteration", "Prompt", "CLIP_Sim", "LPIPS", "RMSE"])
                writer.writerow(["Baseline", best_prompt, best_score, base_metrics['LPIPS'], base_metrics['RMSE']])
            
            # Iteration loop (LLM fine-tuning the prompt)
            for iteration in range(NUM_ITERATIONS):
                print(f"  Iteration {iteration+1}/{NUM_ITERATIONS}...")
                
                candidate_prompts = call_llm_for_prompts(best_prompt, best_score, base_metrics['RMSE'])
                
                for idx, prompt in enumerate(candidate_prompts):
                    gen_img = generator.generate(prompt=prompt, target_filename=filename)
                    metrics = evaluator.evaluate(target_img, gen_img)
                    current_clip = metrics["CLIP_Sim"]
                    
                    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([iteration+1, prompt, current_clip, metrics['LPIPS'], metrics['RMSE']])
                    
                    if current_clip > best_score:
                        best_score = current_clip
                        best_prompt = prompt
                        gen_img.save(os.path.join(run_output_dir, "current_best.png"))

            print(f"-> Winner (Run {run_id}): {best_prompt} (CLIP: {best_score:.4f})")

        extract_top_3_for_report(filename, img_base_dir)
    
    calculate_global_statistics()

if __name__ == "__main__":
    main()