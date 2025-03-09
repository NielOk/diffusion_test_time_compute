import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from inference_experiment_utils import (
    load_code,
    load_model_architecture,
    get_model_paths,
    denoise_to_step,
    classify_generated_images_hf,
    load_hf_classifier,
    HF_MODEL_NAME,
)

def experiment_until_correct(model_type="nlc", target_epoch=100):
    """
    For each digit [0..9]:
      - Sample from Gaussian noise and run a full reverse diffusion (T=999..0).
      - Classify the result. If correct, stop. Otherwise repeat.
    Finally log how many attempts were needed per digit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_ema = True
    use_clip = True
    
    # --------------------
    # 1) Load Model Code
    # --------------------
    (
        TRAINED_MODELS_DIR, 
        _create_mnist_dataloaders,  # not used here
        MNISTDiffusion, 
        ExponentialMovingAverage
    ) = load_code(model_type=model_type)

    # Build the model architecture
    model = load_model_architecture(MNISTDiffusion, device=device, model_type=model_type)
    model_ema = ExponentialMovingAverage(model, decay=0.995, device=device)

    # Grab the sorted model checkpoints
    sorted_model_paths, sorted_epoch_numbers = get_model_paths(TRAINED_MODELS_DIR)

    # Find the .pt file for the specific epoch
    if target_epoch not in sorted_epoch_numbers:
        raise ValueError(f"Epoch {target_epoch} not found in trained checkpoints: {sorted_epoch_numbers}")
    
    idx = sorted_epoch_numbers.index(target_epoch)
    model_path = sorted_model_paths[idx]
    
    # Load checkpoint
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model_ema.load_state_dict(ckpt["model_ema"])

    # Put in eval mode
    model.eval()
    model_ema.eval()

    # ---------------------
    # 2) Load HF Classifier
    # ---------------------
    hf_model, feature_extractor = load_hf_classifier(HF_MODEL_NAME, device=device)
    
    # Our total timesteps (usually 1000)
    T_start = model.timesteps - 1

    # Keep track of how many attempts it takes per digit
    attempts_per_digit = [0]*10

    # For each digit, repeatedly generate until the HF model says it's correct
    for digit in range(10):

        # If label-conditioned, we pass `labels=digit`; otherwise `None`
        labels = None
        if model_type == 'lc':
            labels = torch.full((1,), digit, dtype=torch.long, device=device)

        attempts = 0
        while True:
            attempts += 1
            
            # Random noise: [batch=1, channels=1, height=28, width=28]
            latent = torch.randn(
                (1, model.in_channels, model.image_size, model.image_size),
                device=device
            )

            # Reverse diffusion from T=999 down to 0
            if use_ema:
                final_images = denoise_to_step(
                    model_ema,
                    latent,
                    t=0,
                    start_point=T_start,
                    labels=labels,
                    model_type=model_type,
                    device=device,
                    ema=True,
                    use_clip=use_clip
                )
            else:
                final_images = denoise_to_step(
                    model,
                    latent,
                    t=0,
                    start_point=T_start,
                    labels=labels,
                    model_type=model_type,
                    device=device,
                    ema=False,
                    use_clip=use_clip
                )

            # Classify the resulting image
            predicted_labels = classify_generated_images_hf(final_images, hf_model, feature_extractor, device=device)
            predicted_digit = predicted_labels[0]

            if predicted_digit == digit:
                # If correct, break the loop
                print(f"Digit {digit} succeeded after {attempts} attempt(s).")
                attempts_per_digit[digit] = attempts
                break

    # Final report: how many attempts needed per digit
    print("\n=== Results: Number of Attempts per Digit ===")
    for d in range(10):
        print(f"Digit {d} -> {attempts_per_digit[d]} attempt(s).")

if __name__ == "__main__":
    # Example usage for label-conditioned or non-label-conditioned.
    # 1) Label-conditioned
    experiment_until_correct(model_type="nlc", target_epoch=100)

    # 2) Non-label-conditioned (uncomment to run after the first):
    # experiment_until_correct(model_type="nlc", target_epoch=100)