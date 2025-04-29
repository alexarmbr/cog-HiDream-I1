import tempfile
import torch
import gc
import os
import time
from pathlib import Path
from typing import Literal
from cog import BasePredictor, Input, Path
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Model configs
MODEL_PREFIX = "HiDream-ai"
LLAMA_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowMatchEulerDiscreteScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlowMatchEulerDiscreteScheduler
    }
}

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Load default model (fast)
        self.model_type = "fast"
        self._load_model(self.model_type)
        print("Model loaded successfully!")

    def _load_model(self, model_type):
        """Helper function to load models based on model type"""
        self.config = MODEL_CONFIGS[model_type]
        pretrained_model_name_or_path = self.config["path"]
        
        # Initialize scheduler with correct parameters
        scheduler = self.config["scheduler"](
            num_train_timesteps=1000,
            shift=self.config["shift"],
            use_dynamic_shifting=False
        )
        
        # Load tokenizer and text encoder if not already loaded
        if not hasattr(self, 'tokenizer_4'):
            self.tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
                LLAMA_MODEL_NAME,
                use_fast=False
            )
            
        if not hasattr(self, 'text_encoder_4'):
            self.text_encoder_4 = LlamaForCausalLM.from_pretrained(
                LLAMA_MODEL_NAME,
                output_hidden_states=True,
                output_attentions=True,
                torch_dtype=torch.bfloat16
            ).to("cuda")

        # Load transformer and pipeline
        self.transformer = HiDreamImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16
        ).to("cuda")

        self.pipe = HiDreamImagePipeline.from_pretrained(
            pretrained_model_name_or_path, 
            scheduler=scheduler,
            tokenizer_4=self.tokenizer_4,
            text_encoder_4=self.text_encoder_4,
            torch_dtype=torch.bfloat16
        ).to("cuda", torch.bfloat16)
        self.pipe.transformer = self.transformer

    def _switch_pipe(self, model_type):
        """Switch to a different model type, properly handling memory cleanup"""
        # Move models to CPU and delete them
        if hasattr(self, 'transformer'):
            self.transformer.to('cpu')
            del self.transformer
        if hasattr(self, 'pipe'):
            self.pipe.to('cpu')
            del self.pipe
        if hasattr(self, 'config'):
            del self.config
            
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        self._load_model(model_type)

    def _generate_output_path(self, prompt: str, seed: int, format: Literal["png", "webp"] = "png") -> Path:
        """
        Generate an output path based on the prompt and seed.
        
        Args:
            prompt: The input prompt
            seed: The random seed used
            format: The output image format (png or webp)
            
        Returns:
            Path: The full path where the image should be saved
        """
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get first three words from prompt, joined by underscores
        words = prompt.split()[:3]
        filename_base = "_".join(words)
        
        # Create filename with seed and format
        filename = f"{filename_base}_{seed}.{format}"
        
        return output_dir / filename

    def predict(
        self,
        prompt: str = Input(description="The prompt to generate an image from"),
        resolution: str = Input(
            description="Resolution for the generated image",
            choices=RESOLUTION_OPTIONS,
            default="1024 × 1024 (Square)",
        ),
        seed: int = Input(description="Random seed for reproducibility", default=-1),
        model_type: str = Input(description="Model type to use", choices=list(MODEL_CONFIGS.keys()), default="fast"),
        output_format: str = Input(description="Output image format", choices=["png", "webp"], default="png"),
    ) -> Path:
        """Run a single prediction on the model"""
        # Load different model if requested
        if model_type != self.model_type:
            self._switch_pipe(model_type)
            self.model_type = model_type

        # Set up generator for reproducibility
        if seed == -1:
            seed = torch.randint(0, 1000000, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)

        # Get width and height from resolution
        width, height = parse_resolution(resolution)
        num_inference_steps = self.config["num_inference_steps"]
        guidance_scale = self.config["guidance_scale"]

        # Generate image
        t0 = time.time()
        # with torch.no_grad():
        images = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
        ).images
        t1 = time.time()
        print(f"Time taken: {t1 - t0} seconds")

        # Generate output path
        output_path = self._generate_output_path(prompt, seed, output_format)
        
        # Save output
        if output_format == "png":
            images[0].save(output_path)
        elif output_format == "webp":
            images[0].save(output_path, quality=80, optimize=True)
        return output_path