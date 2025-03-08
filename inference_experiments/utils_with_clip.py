import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
from torchvision import transforms
import clip
from PIL import Image

CHECKPOINT = "epoch_100_steps_00046900.pt"  # Path to your diffusion model checkpoint
DIGIT = 0  # Change this to any digit 0-9

N_CANDIDATES = 32         # Increase candidate pool
SCORING_TIMESTEPS = 50    # Prune every 50 steps
CHECKPOINT_INTERVAL = SCORING_TIMESTEPS
N_EXPERIMENTS = 2         # Number of runs

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')
sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
sys.path.append(TRAINED_MODELS_DIR)

from model import MNISTDiffusion  # Adjust import path as needed

def load_clip_model(device="cuda"):
    """
    Loads CLIP (e.g., RN101) and creates a text embedding based on a descriptive prompt.
    Returns: (clip_model, preprocess, text_embedding)
    """
    clip_model, preprocess = clip.load("RN101", device=device)
    clip_model.eval()

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
    Converts a batch of candidates with shape (B, 1, 28, 28) in [-1, 1] to
    (B, 3, 224, 224) suitable for CLIP scoring.
    """
    batch_images = []
    for img_tensor in candidates:
        # Scale from [-1,1] to [0,1]
        img_tensor = (img_tensor * 0.5) + 0.5

        # Convert to PIL image (grayscale)
        img_pil = transforms.ToPILImage()(img_tensor.cpu())

        # Create an RGB image by pasting the grayscale into all channels
        img_rgb = Image.new("RGB", img_pil.size)
        img_rgb.paste(img_pil)
        
        # Apply the CLIP transform
        clip_ready = clip_input_transform(img_rgb)
        batch_images.append(clip_ready)

    return torch.stack(batch_images, dim=0)

def load_model(ckpt_path, device="cuda"):
    """
    Loads the pre-trained MNISTDiffusion model from the checkpoint.
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

@torch.no_grad()
def reverse_diffusion_search(model, clip_model, text_embedding, n_candidates=N_CANDIDATES,
                             checkpoint_interval=CHECKPOINT_INTERVAL, device="cuda"):
    """
    Runs reverse diffusion starting from random noise. At specified intervals,
    candidates are pruned by scoring with CLIP similarity to the text prompt.
    """
    candidates = torch.randn(
        (n_candidates, model.in_channels, model.image_size, model.image_size),
        device=device
    )
    print(f"[INFO] Beginning reverse diffusion with {n_candidates} initial candidates.")

    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        # Prune at checkpoint intervals
        if t % checkpoint_interval == 0:
            clip_inputs = candidates_to_clip_batch(candidates).to(device)
            image_embeddings = clip_model.encode_image(clip_inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            scores = (image_embeddings @ text_embedding.T).squeeze(dim=-1)

            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores, k=k, largest=True).indices
            candidates = candidates[topk_indices]
            print(f"[DEBUG] Timestep {t:4d} | Pruning to top {candidates.shape[0]} by CLIP similarity.")

    # Final scoring step
    clip_inputs = candidates_to_clip_batch(candidates).to(device)
    image_embeddings = clip_model.encode_image(clip_inputs)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    scores = (image_embeddings @ text_embedding.T).squeeze(dim=-1)
    best_idx = torch.argmax(scores)
    best_candidate = candidates[best_idx]

    print(f"[INFO] Finished reverse diffusion. Final candidate shape: {best_candidate.shape}")
    return best_candidate

def score_candidates(approach, distribution_data, t, candidates):
    """
    Computes a score for each candidate based on the selected approach.
    Returns a 1D tensor of scores (higher is better).

    For the 'clip' approach, distribution_data should be a tuple:
      (clip_model, text_embedding, device)
    """
    if approach == "mse":
        if t not in distribution_data:
            return None
        target_t = distribution_data[t]  # shape [1,1,28,28]
        target = target_t.expand(candidates.shape[0], -1, -1, -1)
        # Negative MSE as score
        scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
        return scores

    elif approach == "bayes":
        posterior_means, posterior_vars = distribution_data
        if t not in posterior_means:
            return None
        mu_t = posterior_means[t].expand(candidates.shape[0], -1, -1, -1)
        var_t = posterior_vars[t]  # scalar float
        squared_errors = F.mse_loss(candidates, mu_t, reduction='none').mean(dim=[1,2,3])
        scores = -squared_errors / (2.0 * var_t)
        return scores

    elif approach == "mixture":
        if t not in distribution_data:
            return None
        mus, var_t = distribution_data[t]  # mus: [N,1,28,28], var_t: float
        c_expanded = candidates.unsqueeze(1)  # [batch,1,1,28,28]
        m_expanded = mus.unsqueeze(0)         # [1,N,1,28,28]
        diff = c_expanded - m_expanded
        sq_dist = diff.pow(2).mean(dim=[2,3,4])  # [batch, N]
        exponent = -sq_dist / (2.0 * var_t)
        max_exponent, _ = exponent.max(dim=1, keepdim=True)
        exponent_shifted = exponent - max_exponent
        sum_exp = torch.exp(exponent_shifted).sum(dim=1)
        mixture_sum = (1.0 / mus.shape[0]) * torch.exp(max_exponent.squeeze()) * sum_exp
        ll = torch.log(mixture_sum + 1e-12)
        return ll

    elif approach == "clip":
        # For the 'clip' approach, distribution_data should be (clip_model, text_embedding, device)
        clip_model, text_embedding, device = distribution_data
        clip_inputs = candidates_to_clip_batch(candidates).to(device)
        with torch.no_grad():
            image_embeddings = clip_model.encode_image(clip_inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        scores = (image_embeddings @ text_embedding.T).squeeze(dim=-1)
        return scores

    else:
        raise ValueError(f"Unknown approach: {approach}")


def plot_image(image_tensor, title="Generated Image"):
    """
    Plots a single diffusion output image (assumed shape: [1, 28, 28] in [-1,1]).
    """
    image = (image_tensor.squeeze().detach().cpu().numpy() + 1) / 2.0
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, CHECKPOINT)

    # Load diffusion model
    model = load_model(ckpt_path, device)

    # Load CLIP model and get text embedding
    clip_model, preprocess, text_embedding = load_clip_model(device)

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

    # Plot the results
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
