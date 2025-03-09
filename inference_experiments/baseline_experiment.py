import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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

def plot_confusion_matrix(cm, classes):
    """
    Minimal utility function to plot a confusion matrix using Matplotlib.
    (No seaborn usage.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Write the counts in each cell
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def experiment_50_images_per_class(model_type="lc", target_epoch=100, num_samples_per_digit=50):
    """
    1) Loads the diffusion model for either 'lc' or 'nlc' at a given epoch (default=100).
    2) Uses denoise_to_step to generate 50 images per digit [0..9].
       - For 'lc', we pass digit labels to the reverse diffusion steps.
       - For 'nlc', we pass None.
    3) Classifies generated images with the HF MNIST model.
    4) Prints per-digit accuracy and shows a confusion matrix.
    """
    # -------------------
    # Hyperparameters
    # -------------------
    use_ema = True            # whether to use the EMA weights
    use_clip = True           # whether to apply clipping
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --------------------
    # 1) Load Model Code
    # --------------------
    (
        TRAINED_MODELS_DIR, 
        _create_mnist_dataloaders,   # not used in this example
        MNISTDiffusion, 
        ExponentialMovingAverage
    ) = load_code(model_type=model_type)

    # Build the model architecture
    model = load_model_architecture(MNISTDiffusion, device=device, model_type=model_type)

    # Build an EMA wrapper
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
    # 2) Generate samples
    # ---------------------
    # Load the HF classifier
    hf_model, feature_extractor = load_hf_classifier(HF_MODEL_NAME, device=device)
    
    # We'll track predictions (from HF) vs. the ground-truth digit
    all_preds = []
    all_labels = []

    # Our total timesteps for the model
    T_start = model.timesteps - 1  # typically 999 or 1000 in your code

    # Generate images for each digit
    for digit in range(10):

        # Start from random noise [B, C=1, H=28, W=28]
        latent = torch.randn(
            (num_samples_per_digit, model.in_channels, model.image_size, model.image_size),
            device=device
        )
        
        # If label-conditioned, pass the digit; if non-label-conditioned, pass None
        if model_type == 'lc':
            labels = torch.full(
                (num_samples_per_digit,), digit, dtype=torch.long, device=device
            )
        else:
            labels = None

        # Reverse diffusion from T=999 down to t=0
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
        
        # -------------------------
        # 3) Classify the images
        # -------------------------
        predicted_labels = classify_generated_images_hf(
            final_images, hf_model, feature_extractor, device=device
        )

        # Keep track for confusion matrix
        all_preds.extend(predicted_labels.tolist())
        all_labels.extend([digit]*num_samples_per_digit)

    # -------------------------
    # 4) Evaluate performance
    # -------------------------
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Per-digit accuracy
    print(f"\n=== {model_type.upper()} Model @ epoch {target_epoch} ===")
    for digit in range(10):
        mask = (all_labels == digit)
        correct = (all_preds[mask] == digit).sum()
        total = mask.sum()
        print(f"Digit {digit}: {correct}/{total} correct, accuracy = {correct/total:.1%}")

    # Overall accuracy
    overall_acc = (all_preds == all_labels).mean()
    print(f"\nOverall accuracy: {overall_acc:.1%}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, [str(i) for i in range(10)])


if __name__ == "__main__":
    # Example usage:
    # 1) Label-conditioned:
    experiment_50_images_per_class(model_type="lc", target_epoch=100, num_samples_per_digit=50)
    
    # 2) Non-label-conditioned (uncomment to run immediately afterwards)
    # experiment_50_images_per_class(model_type="nlc", target_epoch=100, num_samples_per_digit=50)
