from predict import Predictor
from pathlib import Path
import time

# TODO:
# push weights to google cloud
# turn taylor cache on/off with a flag
# push to replicate

def main():
    # Create predictor instance
    predictor = Predictor()
    
    # Call setup to load the model
    predictor.setup()

    PROMPT = "a rainbow colored grizzly bear howling at the moon, realistic"

    # Call predict with all default arguments explicitly specified
    output_path = predictor.predict(
        prompt=PROMPT,  # Empty prompt as default
        seed=567,  # Default random seed
        resolution="1024 × 1024 (Square)",
        model_type="fast",  # Default model type
        output_format="png"  # Default output format
    )

    t0 = time.time()
    for i in range(3):
        go_fast = i % 2 == 0
        print(f"go_fast: {go_fast}")
        output_path = predictor.predict(
            prompt=PROMPT,  # Empty prompt as default
            seed=-1,  # Default random seed
            resolution="1024 × 1024 (Square)",
            model_type="fast",  # Default model type
            output_format="png",
            go_fast=go_fast
        )
    t1 = time.time()
    print(f"avg time per image: {(t1 - t0) / 3} seconds")

if __name__ == "__main__":
    main() 