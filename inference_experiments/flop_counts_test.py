import torch.profiler as profiler
import torch

from inference_experiment_utils import *

def test_flops():
    model_type = "lc"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_candidates = 3
    digit_to_generate = 8
    t = 1

    # Load code
    TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage = load_code(model_type=model_type)

    # Load model architecture
    model = load_model_architecture(MNISTDiffusion, device=device, model_type=model_type)
    model_ema = ExponentialMovingAverage(model, decay=0.995, device=device)

    # Get model filepaths
    sorted_model_paths, sorted_epoch_numbers = get_model_paths(TRAINED_MODELS_DIR)

    # Select model to load based on epoch number
    epoch_number = 100
    model_to_load = sorted_model_paths[sorted_epoch_numbers.index(epoch_number)]

    # Load model weights
    checkpoint = torch.load(model_to_load, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])

    model.eval()
    model_ema.eval()

    # Load data
    train_loader, test_loader = create_mnist_dataloaders(batch_size=128,image_size=28)

    candidates = torch.randn(
        (n_candidates, model.in_channels, model.image_size, model.image_size),
        device=device
    )
    noise = torch.randn_like(candidates)
    t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
    if model_type == 'lc':
        labels = torch.full((candidates.shape[0],), digit_to_generate, dtype=torch.long, device=device)
    else:
        labels = None

    if model_type == 'lc':
        reverse_diffusion_input = (candidates, t_tensor, noise, labels)
        forward_diffusion_input = (candidates, t_tensor, noise)
    else:
        reverse_diffusion_input = (candidates, t_tensor, noise)
        forward_diffusion_input = (candidates, t_tensor, noise)

    # Run flops test for reverse diffusion
    with profiler.profile(with_flops=True) as prof:
        model._reverse_diffusion(*reverse_diffusion_input)
    
    # Sum all flops
    reverse_diffusion_total_flops = sum(event.flops for event in prof.key_averages())
    print(f"Total FLOPs per reverse diffusion for n_candidates={n_candidates}: {reverse_diffusion_total_flops}")

    # Run flops test for forward diffusion
    with profiler.profile(with_flops=True) as prof:
        model._forward_diffusion(*forward_diffusion_input)
    
    # Sum all flops
    total_flops = sum(event.flops for event in prof.key_averages())
    print(f"Total FLOPs per forward diffusion for n_candidates={n_candidates}: {total_flops}")

if __name__ == '__main__':
    test_flops()