import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools

# === Import the required helpers from your snippet ===
from inference_experiment_utils import (
    load_code,
    load_model_architecture,
    get_model_paths,
    denoise_to_step,
    classify_generated_images_hf,
    load_hf_classifier,
    HF_MODEL_NAME,
)

def plot_class_distribution(labels, possible_labels, title):
    counts, _ = np.histogram(labels, bins=np.arange(11))  # Bin edges from 0 to 10
    
    plt.figure()
    plt.bar(possible_labels, counts, width=0.8, align='center', edgecolor='black')
    plt.xticks(possible_labels)  # Ensure all class labels (0-9) are displayed
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.savefig(f"{title}.png")

def nlc_experiment_multiple_images(target_epoch=100, num_samples=450, batch_size=50):

    model_type = "nlc"
    use_ema = True
    use_clip = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        TRAINED_MODELS_DIR, 
        _create_mnist_dataloaders,   # not used in this example
        MNISTDiffusion, 
        ExponentialMovingAverage
    ) = load_code(model_type=model_type)

    model = load_model_architecture(MNISTDiffusion, device=device, model_type=model_type)
    model_ema = ExponentialMovingAverage(model, decay=0.995, device=device)

    sorted_model_paths, sorted_epoch_numbers = get_model_paths(TRAINED_MODELS_DIR)

    if target_epoch not in sorted_epoch_numbers:
        raise ValueError(f"Epoch {target_epoch} not found in trained checkpoints: {sorted_epoch_numbers}")

    idx = sorted_epoch_numbers.index(target_epoch)
    model_path = sorted_model_paths[idx]

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model_ema.load_state_dict(ckpt["model_ema"])

    # Put in eval mode
    model.eval()
    model_ema.eval()

    hf_model, feature_extractor = load_hf_classifier(HF_MODEL_NAME, device=device)

    print("Models loaded successfully")
    
    # We'll track predictions (from HF) vs. the ground-truth digit
    all_preds = []

    for batch_id in range(num_samples // batch_size):
        print(f"Batch {batch_id}")
        candidates = torch.randn((batch_size, 1, model.in_channels, model.image_size, model.image_size), device=device)
        candidates = candidates.view(batch_size, model.in_channels, model.image_size, model.image_size)
        noise = torch.randn_like(candidates)

        if use_ema:
            if use_clip:
                images = denoise_to_step(model_ema, candidates, t=0, start_point=model.timesteps - 1, labels=None, B=batch_size, K=1,model_type=model_type, device=device, ema=use_ema, use_clip=use_clip)
            else:
                raise NotImplementedError("Not implemented for non-clipped")
        else:
            raise NotImplementedError("Not implemented for non-EMA models")
        
        predicted_labels = classify_generated_images_hf(images, hf_model, feature_extractor, device=device)
        all_preds.extend(predicted_labels.tolist())

    all_preds = np.array(all_preds)
    possible_labels = [i for i in range(10)]
    plot_class_distribution(all_preds, title="Baseline Predicted Class Distribution", possible_labels=possible_labels)

if __name__ == "__main__":

    nlc_experiment_multiple_images(target_epoch=100, num_samples=450, batch_size=50)
