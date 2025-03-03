import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torchvision import transforms
from torch.utils.data import DataLoader
import clip
from PIL import Image

# ============================================================================
# Constants (tweak these as needed)
# ============================================================================
CHECKPOINT = "epoch_100_steps_00046900.pt"
DIGIT = 0 # Change this to any digit 0-9

# Increase candidate pool
N_CANDIDATES = 32  # was 16

# More frequent pruning
SCORING_TIMESTEPS = 50  # prune every 50 steps
CHECKPOINT_INTERVAL = SCORING_TIMESTEPS
N_EXPERIMENTS = 2  # number of runs

# Directories and model import
MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')
sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
sys.path.append(TRAINED_MODELS_DIR)

from model import MNISTDiffusion  # Adjust import path if needed


# ============================================================================
# 1. Load CLIP and prepare text embedding
# ============================================================================
def load_clip_model(device="cuda"):
    """
    Loads OpenAI CLIP (ViT-B/32) and a textual prompt describing the digit.
    Returns (clip_model, clip_preprocess, text_embedding).
    """
    clip_model, preprocess = clip.load("RN101", device=device)
    clip_model.eval()

    # A more descriptive text prompt:
    text_prompt = (
        f"An accurately drawn, neatly handwritten digit '{DIGIT}' in the style "
        f"of MNIST, on a clean white background, high contrast, no artifacts, black ink."
    )

    with torch.no_grad():
        text_tokens = clip.tokenize([text_prompt]).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    print(f"[INFO] Using text prompt: \"{text_prompt}\"")
    return clip_model, preprocess, text_embedding


# ============================================================================
# 2. Convert diffusion outputs to a format suitable for CLIP scoring
# ============================================================================
# This transform expects a PIL image (3-channel) and normalizes for CLIP.
clip_input_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

def candidates_to_clip_batch(candidates):
    """
    Converts a batch of shape (B, 1, 28, 28) in [-1, 1] range
    to (B, 3, 224, 224) in CLIP’s normalized range.
    We'll manually create a PIL image and replicate channels.
    """
    batch_images = []
    for img_tensor in candidates:
        # Scale from [-1,1] to [0,1]
        img_tensor = (img_tensor * 0.5) + 0.5

        # Convert single-channel to PIL grayscale
        img_pil = transforms.ToPILImage()(img_tensor.cpu())

        # Create an RGB image so that it has 3 channels
        img_rgb = Image.new("RGB", img_pil.size)
        img_rgb.paste(img_pil)  # Paste the grayscale into all channels

        # Now pass the PIL image to the transform pipeline
        clip_ready = clip_input_transform(img_rgb)
        batch_images.append(clip_ready)

    return torch.stack(batch_images, dim=0)


# ============================================================================
# 3. Load your MNIST Diffusion model
# ============================================================================
def load_model(ckpt_path, device="cuda"):
    """
    Loads the pre-trained MNISTDiffusion model from checkpoint.
    """
    model = MNISTDiffusion(
        timesteps=1000,
        image_size=28,
        in_channels=1,
        base_dim=64,
        dim_mults=[2, 4]
    )
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


# ============================================================================
# 4. Reverse diffusion with CLIP-based scoring
# ============================================================================
@torch.no_grad()
def reverse_diffusion_search(
    model,
    clip_model,
    text_embedding,
    n_candidates=N_CANDIDATES,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    device="cuda"
):
    """
    Generates samples from the diffusion model in reverse,
    pruning half the batch periodically by highest CLIP similarity to the text prompt.
    """
    # Start with random noise
    candidates = torch.randn(
        (n_candidates, model.in_channels, model.image_size, model.image_size),
        device=device
    )

    print(f"[INFO] Beginning reverse diffusion with {n_candidates} initial candidates.")

    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)

        # Single step of reverse diffusion
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        # Prune at certain intervals
        if t % checkpoint_interval == 0:
            # 1. Convert to CLIP format
            clip_inputs = candidates_to_clip_batch(candidates)
            clip_inputs = clip_inputs.to(device)

            # 2. Get CLIP embeddings
            image_embeddings = clip_model.encode_image(clip_inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            # 3. Compute similarity
            scores = (image_embeddings @ text_embedding.T).squeeze(dim=-1)

            # 4. Keep top half (or at least 1)
            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores, k=k, largest=True).indices
            candidates = candidates[topk_indices]

            print(f"[DEBUG] Timestep {t:4d} | Pruning to top {candidates.shape[0]} by CLIP similarity.")
    
    # Final scoring step
    clip_inputs = candidates_to_clip_batch(candidates)
    clip_inputs = clip_inputs.to(device)
    image_embeddings = clip_model.encode_image(clip_inputs)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

    scores = (image_embeddings @ text_embedding.T).squeeze(dim=-1)
    best_idx = torch.argmax(scores)
    best_candidate = candidates[best_idx]

    print(f"[INFO] Finished reverse diffusion. Final candidate shape: {best_candidate.shape}")
    return best_candidate


# ============================================================================
# 5. Plotting utility
# ============================================================================
def plot_image(image_tensor, title="Generated Image"):
    """
    Utility to plot a single diffusion output image in grayscale.
    Assumes image_tensor is shape (1, 28, 28) in [-1,1].
    """
    image = (image_tensor.squeeze().detach().cpu().numpy() + 1) / 2.0
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')


# ============================================================================
# 6. Main
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, CHECKPOINT)

    # 6.1 Load your diffusion model
    model = load_model(ckpt_path, device)

    # 6.2 Load CLIP model + text embedding
    clip_model, preprocess, text_embedding = load_clip_model(device)

    # 6.3 Run multiple experiments
    outputs = []
    for i in range(N_EXPERIMENTS):
        best_candidate = reverse_diffusion_search(
            model=model,
            clip_model=clip_model,
            text_embedding=text_embedding,
            device=device
        )
        outputs.append(best_candidate)
        print(f"[INFO] Experiment {i+1} complete.")

    # 6.4 Plot results
    num_cols = 5
    num_rows = (N_EXPERIMENTS + num_cols - 1) // num_cols
    plt.figure(figsize=(num_cols * 3, num_rows * 3))

    for idx, out in enumerate(outputs):
        plt.subplot(num_rows, num_cols, idx + 1)
        plot_image(out, title=f"Run {idx+1}")
    plt.suptitle(f"Reverse Diffusion Outputs (CLIP-scored) for Digit {DIGIT}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

