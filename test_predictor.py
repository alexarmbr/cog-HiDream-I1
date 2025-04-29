from predict import Predictor
from pathlib import Path
import time
def main():
    # Create predictor instance
    predictor = Predictor()
    
    # Call setup to load the model
    predictor.setup()
    
    # Call predict with all default arguments explicitly specified
    output_path = predictor.predict(
        prompt="A highly detailed image of a rainbow colored grizzly bear, roaring ferociously at the moon",  # Empty prompt as default
        aspect_ratio="1:1",  # Default aspect ratio
        seed=-1,  # Default random seed
        resolution="1024 × 1024 (Square)",
        model_type="fast",  # Default model type
        output_format="png"  # Default output format
    )

    t0 = time.time()
    for i in range(3):
        output_path = predictor.predict(
            prompt="A highly detailed image of a rainbow colored grizzly bear, roaring ferociously at the moon",  # Empty prompt as default
            aspect_ratio="1:1",  # Default aspect ratio
            seed=-1,  # Default random seed
            resolution="1024 × 1024 (Square)",
            model_type="fast",  # Default model type
            output_format="webp"  # Default output format
        )
    t1 = time.time()
    print(f"avg time per image: {(t1 - t0) / 3} seconds")

if __name__ == "__main__":
    main() 