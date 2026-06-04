import os
from pathlib import Path
from PIL import Image, ImageChops

import utils

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_FILENAME = "1159_7.png"
SAMPLE_PROMPT = "there is a small hedgehog sitting on top of a block of cake"


def load_sample_target():
    target_path = DATA_DIR / SAMPLE_FILENAME
    if not target_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_path}")
    return Image.open(target_path).convert("RGB"), SAMPLE_FILENAME


def images_equal(img1: Image.Image, img2: Image.Image) -> bool:
    diff = ImageChops.difference(img1, img2)
    return not diff.getbbox()


def run_generation_test(prompt: str, filename: str):
    from generator import LCMGenerator
    generator, _ = utils.init_generator_evaluator(LCMGenerator, None)
    image = generator.generate(prompt=prompt, target_filename=filename)
    return image


def run_evaluation_test(target_img: Image.Image, gen_img: Image.Image):
    from evaluator import ImageEvaluator
    _, evaluator = utils.init_generator_evaluator(None, ImageEvaluator)
    return evaluator.evaluate(target_img, gen_img)


def run_determinism_test(prompt: str, filename: str):
    from generator import LCMGenerator
    generator, _ = utils.init_generator_evaluator(LCMGenerator, None)
    img1 = generator.generate(prompt=prompt, target_filename=filename)
    img2 = generator.generate(prompt=prompt, target_filename=filename)
    same = images_equal(img1, img2)
    return same, img1, img2


def main():
    print("Pipeline validation script")
    print("---------------------------")

    target_img, filename = load_sample_target()
    print(f"Sample target: {filename}")
    print(f"Sample prompt: {SAMPLE_PROMPT}\n")

    print("1) Generating test image...")
    generated_img = run_generation_test(SAMPLE_PROMPT, filename)
    utils.safe_save_image(generated_img, BASE_DIR / "outputs" / "validate_sample.png")
    print(f"  -> Image saved to outputs/validate_sample.png")

    print("2) Evaluating metrics...")
    metrics = run_evaluation_test(target_img, generated_img)
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")

    print("3) Testing generation determinism...")
    same, img1, img2 = run_determinism_test(SAMPLE_PROMPT, filename)
    print(f"  Deterministic generation: {'yes' if same else 'no'}")
    if not same:
        diff_path = BASE_DIR / "outputs" / "validate_difference.png"
        ImageChops.difference(img1, img2).save(diff_path)
        print(f"  Difference image saved to outputs/validate_difference.png")

    if os.environ.get("GROQ_API_KEY"):
        try:
            print("\n4) GROQ API key detected. Testing prompt variation generation...")
            client = utils.init_groq_client()
            variations = utils.call_llm_for_prompts(client, SAMPLE_PROMPT, clip_score=0.0, rmse_score=0.0)
            print(f"  Prompt variations: {variations}")
        except Exception as exc:
            print(f"  Failed to test GROQ/LLM: {exc}")
    else:
        print("\n4) GROQ_API_KEY not configured. Skipping LLM test.")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
