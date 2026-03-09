import torch
import torch.nn as nn
import os
import time
from src.utils.UtilsPlot import plot_metrics

# ----------------------------
# TRAINING EPOCH
# ----------------------------
def train_epoch(model, optimizer, criterion, train_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    total_loss = 0.0
    nb_ite = 0

    for _, (O, L, _) in enumerate(train_loader):
        optimizer.zero_grad()
        batch_loss = 0.0

        for i in range(len(O)):
            original_true = O[i].to(device)
            low_resolution = L[i].to(device)

            # Normalisation des dimensions
            if original_true.dim() == 2:
                original_true = original_true.unsqueeze(0).unsqueeze(0)
            elif original_true.dim() == 3:
                original_true = original_true.unsqueeze(0)

            if low_resolution.dim() == 2:
                low_resolution = low_resolution.unsqueeze(0).unsqueeze(0)
            elif low_resolution.dim() == 3:
                low_resolution = low_resolution.unsqueeze(0)

            # Calcul des facteurs de décimation
            decim_row = original_true.shape[2] // low_resolution.shape[2]
            decim_col = original_true.shape[3] // low_resolution.shape[3]

            # Forward pass
            original_pred = model(low_resolution, decim_row=decim_row, decim_col=decim_col)

            # Calcul de la perte
            loss = criterion(original_pred, original_true)
            batch_loss += loss
            total_loss += loss.item()
            nb_ite += 1

            # Libération mémoire
            del original_true, low_resolution, original_pred, loss
            torch.cuda.empty_cache()

        # Backprop
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        del batch_loss
        torch.cuda.empty_cache()

    avg_train_loss = total_loss / nb_ite
    return avg_train_loss


def validation_epoch(model, criterion, validation_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    total_loss = 0.0
    nb_ite = 0

    with torch.no_grad():
        for _, (O, L, _) in enumerate(validation_loader):
            for i in range(len(O)):
                original_true = O[i].to(device)
                low_resolution = L[i].to(device)

                # Normalisation des dimensions
                if original_true.dim() == 2:
                    original_true = original_true.unsqueeze(0).unsqueeze(0)
                elif original_true.dim() == 3:
                    original_true = original_true.unsqueeze(0)

                if low_resolution.dim() == 2:
                    low_resolution = low_resolution.unsqueeze(0).unsqueeze(0)
                elif low_resolution.dim() == 3:
                    low_resolution = low_resolution.unsqueeze(0)

                # Décimation
                decim_row = original_true.shape[2] // low_resolution.shape[2]
                decim_col = original_true.shape[3] // low_resolution.shape[3]

                # Forward pass
                original_pred = model(low_resolution, decim_row=decim_row, decim_col=decim_col)

                # Calcul de la perte
                loss = criterion(original_pred, original_true)
                total_loss += loss.item()
                nb_ite += 1

                del original_true, low_resolution, original_pred, loss
            torch.cuda.empty_cache()

    avg_validation_loss = total_loss / nb_ite
    return avg_validation_loss

# ----------------------------
# EARLY STOPPING
# ----------------------------
def early_stop(best_validation_loss, avg_validation_loss, epoch_no_improve, min_delta=0.0):
    if avg_validation_loss + min_delta < best_validation_loss:
        best_validation_loss = avg_validation_loss
        epoch_no_improve = 0
    else:
        epoch_no_improve += 1
    return best_validation_loss, epoch_no_improve

# ----------------------------
# MAIN TRAIN FUNCTION
# ----------------------------
def train(model, optimizer, criterion, train_loader, validation_loader, 
          nb_epoch, patience, output_dir, decim_row=1, decim_col=1, min_delta=0.0):
    
    start_time_total = time.time()
    best_validation_loss = float("inf")
    epoch_no_improve = 0
    best_model_state = None
    epoch_save = 0
    metrics = {"train_loss": [], "validation_loss": []}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(nb_epoch):
        epoch_start_time = time.time()
        print(f"\n{'-'*60}")
        print(f"EPOQUE {epoch + 1}/{nb_epoch}")
        print(f"{'-'*60}")

        # Training
        avg_train_loss = train_epoch(model, optimizer, criterion, train_loader)

        # Validation
        avg_validation_loss = validation_epoch(model, criterion, validation_loader)

        # Sauvegarde des métriques
        metrics["train_loss"].append(avg_train_loss)
        metrics["validation_loss"].append(avg_validation_loss)

        # Affichage
        epoch_duration = time.time() - epoch_start_time
        print(f"Resultats:")
        print(f" Train Loss: {avg_train_loss:.6f}")
        print(f" Validation Loss: {avg_validation_loss:.6f}")

        # Early stopping
        previous_best = best_validation_loss
        best_validation_loss, epoch_no_improve = early_stop(
            best_validation_loss, avg_validation_loss, epoch_no_improve, min_delta
        )

        print(f"\nEarly Stopping:")
        print(f"  epoques sans amelioration: {epoch_no_improve}/{patience}")
        print(f"  Meilleure validation loss: {best_validation_loss:.6f}")

        if epoch_no_improve == 0:
            improvement = previous_best - best_validation_loss
            print(f" Nouveau meilleur modele! (amelioration: {improvement:.6f})")
            best_model_state = model.state_dict()
            epoch_save = epoch + 1

        if epoch_no_improve >= patience:
            print(f"Arret anticipe a lepoque {epoch + 1}")
            break

    total_duration = time.time() - start_time_total
    torch.save(best_model_state, os.path.join(output_dir, "best_model.pth"))
    print(f"[FIN] Entrainement termine en {total_duration:.2f}s")
    print(f"[SAVE] Meilleur modele sauvegarde dans {os.path.join(output_dir, 'best_model.pth')} a epoque {epoch_save}")

    # Trace les courbes
    plot_metrics(metrics, output_dir)
