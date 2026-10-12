import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision.transforms import v2

# Correct import
from model import Model  
from data_loader import PolyGen
from freq_space_interpolation import freq_space_interpolation, extract_amp_spectrum


# -------------------------------------------------------------------------
# FIXED LOCAL TRAINING (Algorithm 2 inside Algorithm 1)
# -------------------------------------------------------------------------
def local_train_with_synth(model, local_ds, other_datasets, device, round_num, center_id,
                           synth_epochs=1, real_epochs=1, save_synth_images=True, save_dir="synthetic_vis"):

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Precompute amplitudes from other centers
    target_amplitudes = []
    for ds in other_datasets:
        for _ in range(min(50, len(ds))):
            idx = random.randint(0, len(ds)-1)
            img, _ = ds[idx]
            target_amplitudes.append(extract_amp_spectrum(img))

    # Prepare directories
    if save_synth_images:
        os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # SYNTHETIC TRAINING (Algorithm 2)
    # ---------------------------------------------------------------------
    model.train()
    for ep in range(synth_epochs):
        for idx in range(len(local_ds)):
            img, mask = local_ds[idx]

            # --- mask filename fix (no "]") ---
            mask = mask[:,:,:]  # identity, ensures no detached path later

            # Pick random target spectrum
            target_amp = random.choice(target_amplitudes)
            lam = random.uniform(0, 1)

            # Create synthetic image
            x_hat = freq_space_interpolation(img, target_amp, ratio=lam)
            
            # Ensure synthetic image has same shape as input
            if x_hat.shape != img.shape:
                x_hat = x_hat.view(img.shape)

            # Save synthetic preview
            if save_synth_images and idx < 3 and ep == 0:
                save_image(x_hat, os.path.join(
                    save_dir,
                    f"round{round_num}_center{center_id}_img{idx}_synth.jpg"
                ))

            # Train on synthetic
            x_hat = x_hat.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            optimizer.zero_grad()
            out = model(x_hat)
            loss = criterion(out.squeeze(1), mask.squeeze(1))
            loss.backward()
            optimizer.step()

    # ---------------------------------------------------------------------
    # REAL TRAINING
    # ---------------------------------------------------------------------
    for ep in range(real_epochs):
        for idx in range(len(local_ds)):
            img, mask = local_ds[idx]
            img = img.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out.squeeze(1), mask.squeeze(1))
            loss.backward()
            optimizer.step()

    return model


# -------------------------------------------------------------------------
# FEDERATED AGGREGATION
# -------------------------------------------------------------------------
def federated_aggregate(local_states, weights):
    new_state = {}
    keys = local_states[0].keys()
    for k in keys:
        new_state[k] = sum(w * s[k] for s, w in zip(local_states, weights))
    return new_state


# -------------------------------------------------------------------------
# FEDERATED TRAINING LOOP (Algorithm 1 + Algorithm 2 inside)
# -------------------------------------------------------------------------
def run_federated_training(centers, data_root, rounds=6, device=None, model_save_dir="saved_models"):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(model_save_dir, exist_ok=True)

    # Load all centers
    datasets = [PolyGen(root=data_root, center=c, transform=v2.Compose([
        v2.Resize(size=(256, 256)),
        v2.ToDtype(torch.float32, scale=True)
    ])) for c in centers]
    K = len(datasets)
    agg_weights = [1.0 / K] * K

    # Initialize global model
    global_model = Model()
    global_state = global_model.state_dict()

    # -----------------------------------------------------------------
    for t in range(rounds):
        print(f"\n=== Federated Round {t+1}/{rounds} ===")
        local_states = []

        for k in range(K):
            print(f"Client {k}: generating synthetic images + training...")

            local_model = Model()
            local_model.load_state_dict(copy.deepcopy(global_state))

            # datasets except this one
            other_ds = [datasets[j] for j in range(K) if j != k]

            # Run local update + synth training
            local_model = local_train_with_synth(
                local_model, datasets[k], other_ds, device,
                round_num=t+1, center_id=k,
                synth_epochs=1, real_epochs=1,
                save_synth_images=True,
                save_dir=f"synthetic_vis/round{t+1}_center{k}"
            )

            local_states.append(copy.deepcopy(local_model.state_dict()))

        # Federated aggregation
        global_state = federated_aggregate(local_states, agg_weights)
        global_model.load_state_dict(global_state)

        # Save checkpoint
        ckpt_path = os.path.join(model_save_dir, f"global_round_{t+1}.pth")
        torch.save(global_state, ckpt_path)
        print(f"Saved global checkpoint: {ckpt_path}")

    # -----------------------------------------------------------------
    final_path = os.path.join(model_save_dir, "global_final.pth")
    torch.save(global_state, final_path)
    print(f"\nTraining finished. Final global model saved at: {final_path}")

    return final_path


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    centers = [1,2,3,4,5,6]  # all 6 centers
    data_root = "PolypGen2021_MultiCenterData_v3"
    device = "cpu"  # change to "cuda" if GPU works
    run_federated_training(centers, data_root, rounds=6, device=device)