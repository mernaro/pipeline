import os
import json
import time
from pathlib import Path
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Unet import UNet
from src.utils.UtilsPlot import plot_metrics


class PipeTrainer:

    def __init__(
        self,
        unfolding_model,
        unet_dropout = 0.3,
        w_sr         = 1.0,
        w_seg        = 0.0,
        grad_clip    = 1.0,
    ):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.w_sr      = w_sr
        self.w_seg     = w_seg
        self.grad_clip = grad_clip

        self.unfolding = unfolding_model.to(self.device)
        self.unet      = UNet(dropout_rate=unet_dropout).to(self.device)

        n_params = sum(p.numel() for p in self.unet.parameters())
        print(f"[PipeTrainer] device={self.device} | w_sr={w_sr} | w_seg={w_seg}")
        print(f"[PipeTrainer] UNet params : {n_params:,}")


    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _seg_loss(self, seg_pred, seg_gt):
        seg_pred = torch.clamp(seg_pred, 1e-6, 1.0 - 1e-6)
        bce  = F.binary_cross_entropy(seg_pred, seg_gt)
        num  = 2.0 * (seg_pred * seg_gt).sum() + 1e-6
        den  = seg_pred.sum() + seg_gt.sum() + 1e-6
        dice = 1.0 - num / den
        return bce + dice

    def _prepare_seg_gt(self, seg_gt, target_shape):
        gt = seg_gt.float().to(self.device)
        while gt.dim() > 2:
            gt = gt.squeeze(0)
        if gt.shape != target_shape:
            gt = F.interpolate(
                gt.unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='nearest'
            ).squeeze(0).squeeze(0)
        return gt.unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _get_phase(self, epoch, phase1_epochs, phase2_epochs):
        if epoch < phase1_epochs:
            return 1
        elif epoch < phase1_epochs + phase2_epochs:
            return 2
        else:
            return 3

    def _get_w_seg_curriculum(self, epoch, phase1_epochs, phase2_epochs, w_seg_max):
        phase = self._get_phase(epoch, phase1_epochs, phase2_epochs)
        if phase == 1:
            return 0.0
        elif phase == 2:
            return 1.0
        else:
            phase3_epoch = epoch - phase1_epochs - phase2_epochs
            ramp         = min(phase3_epoch / 50.0, 1.0)
            return 0.1 + ramp * (w_seg_max - 0.1)

    def _freeze(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def _unfreeze(self, model):
        for p in model.parameters():
            p.requires_grad = True

    def _reset_lr(self, optimizer, lr):
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    def _make_schedulers(self, optimizer_unfolding, optimizer_unet):
        scheduler_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_unet,
            mode     = "min",
            factor   = 0.2,
            patience = 10,
            min_lr   = 1e-6,
        )
        scheduler_unfolding = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_unfolding,
            mode     = "min",
            factor   = 0.5,
            patience = 15,
            min_lr   = 1e-7,
        )
        return scheduler_unfolding, scheduler_unet

    # ------------------------------------------------------------------
    # Compute losses — PHASE-AWARE
    # BUG FIX : original_pred.detach() en phase 2 pour isoler le UNet
    # BUG FIX : loss_total ne contient que les termes actifs par phase
    # ------------------------------------------------------------------

    def _compute_losses(self, O_i, L_i, S_i, criterion, phase, w_seg_current):
        original_true  = O_i.to(self.device)
        low_resolution = L_i.to(self.device)

        res_size  = original_true.size()
        inp_size  = low_resolution.size()
        decim_row = res_size[0] // inp_size[0]
        decim_col = res_size[1] // inp_size[1]

        assert decim_row > 0 and decim_col > 0, (
            f"Decimation invalide : res={res_size}, inp={inp_size}."
        )

        original_pred = self.unfolding(low_resolution, decim_row, decim_col)
        loss_sr       = criterion(original_pred, original_true)

        # FIX : en phase 2 l'unfolding est gelé ? on detache original_pred
        # pour que le backward du UNet ne remonte pas dans l'unfolding
        if phase == 2:
            unet_input = original_pred.detach()
        else:
            unet_input = original_pred

        seg_pred  = self.unet(unet_input)
        H, W      = seg_pred.shape[2], seg_pred.shape[3]
        seg_gt_4d = self._prepare_seg_gt(S_i, target_shape=(H, W))
        loss_seg  = self._seg_loss(seg_pred, seg_gt_4d)

        # FIX : loss_total ne contient que les termes du modele actif
        if phase == 1:
            loss_total = loss_sr                                         # SR seule
        elif phase == 2:
            loss_total = loss_seg                                        # Seg seule
        else:
            loss_total = self.w_sr * loss_sr + w_seg_current * loss_seg # Joint

        return loss_sr, loss_seg, loss_total

    # ------------------------------------------------------------------
    # Train / Val epochs
    # ------------------------------------------------------------------

    def train_epoch(self, optimizer_unfolding, optimizer_unet, criterion,
                    train_loader, phase, w_seg_current):
        avg_train_loss = 0.0
        avg_seg_loss   = 0.0
        nb_ite         = 0

        self.unfolding.train()
        self.unet.train()

        if phase == 1:
            self._freeze(self.unet)
            self._unfreeze(self.unfolding)
        elif phase == 2:
            self._freeze(self.unfolding)
            self._unfreeze(self.unet)
        else:
            self._unfreeze(self.unfolding)
            self._unfreeze(self.unet)

        for _, (O, L, P, S) in enumerate(train_loader):
            nb_ite += len(O)

            for i in range(len(O)):
                optimizer_unfolding.zero_grad()
                optimizer_unet.zero_grad()

                loss_sr, loss_seg, loss_total = self._compute_losses(
                    O[i], L[i], S[i], criterion, phase, w_seg_current
                )

                loss_total.backward()

                if self.grad_clip > 0:
                    if phase != 2:
                        nn.utils.clip_grad_norm_(
                            self.unfolding.parameters(), max_norm=self.grad_clip
                        )
                    if phase != 1:
                        nn.utils.clip_grad_norm_(
                            self.unet.parameters(), max_norm=self.grad_clip
                        )

                # FIX : en phase 1 ne pas toucher optimizer_unet et vice versa
                if phase in (1, 3):
                    optimizer_unfolding.step()
                if phase in (2, 3):
                    optimizer_unet.step()

                avg_train_loss += loss_sr.item()
                avg_seg_loss   += loss_seg.item()

        avg_train_loss /= nb_ite
        avg_seg_loss   /= nb_ite
        return avg_train_loss, avg_seg_loss

    def validation_epoch(self, criterion, validation_loader, phase, w_seg_current):
        avg_validation_loss = 0.0
        avg_seg_loss        = 0.0
        nb_ite              = 0
        self.unfolding.eval()
        self.unet.eval()

        with torch.no_grad():
            for _, (O, L, P, S) in enumerate(validation_loader):
                nb_ite += len(O)

                for i in range(len(O)):
                    loss_sr, loss_seg, _ = self._compute_losses(
                        O[i], L[i], S[i], criterion, phase, w_seg_current
                    )
                    avg_validation_loss += loss_sr.item()
                    avg_seg_loss        += loss_seg.item()

        avg_validation_loss /= nb_ite
        avg_seg_loss        /= nb_ite
        return avg_validation_loss, avg_seg_loss

    # ------------------------------------------------------------------
    # Early stop
    # ------------------------------------------------------------------

    def early_stop(self, best_loss, current_loss, epoch_no_improve, min_delta):
        if current_loss + min_delta < best_loss:
            best_loss        = current_loss
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1
        return best_loss, epoch_no_improve

    # ------------------------------------------------------------------
    # Train loop principal
    # ------------------------------------------------------------------

    def train(
        self,
        optimizer_unfolding,
        optimizer_unet,
        criterion,
        train_loader,
        batch_size_train,
        validation_loader,
        batch_size_val,
        nb_epoch,
        patience,
        output_dir,
        min_delta,
        phase1_epochs    = 150,
        phase2_epochs    = 150,
        w_seg_max        = 0.3,
        lr_unfolding_p1  = 1e-3,   # LR unfolding phase 1
        lr_unet_p2       = 1e-4,   # LR UNet phase 2
        lr_unfolding_p3  = 1e-4,   # LR unfolding phase 3 (fine-tuning)
        lr_unet_p3       = 1e-5,   # LR UNet phase 3 (fine-tuning)
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        start_time_total     = time.time()
        best_validation_loss = float("inf")
        epoch_no_improve     = 0
        best_model_state     = None
        epoch_save           = 0
        val_history          = deque(maxlen=5)
        current_phase        = 0

        print(f"\n[PHASES] Phase 1 : epoques 1-{phase1_epochs} (SR seule)")
        print(f"[PHASES] Phase 2 : epoques {phase1_epochs+1}-{phase1_epochs+phase2_epochs} (Seg seule)")
        print(f"[PHASES] Phase 3 : epoques {phase1_epochs+phase2_epochs+1}-{nb_epoch} (Fine-tuning conjoint)")

        # Schedulers initialises pour phase 1
        scheduler_unfolding, scheduler_unet = self._make_schedulers(
            optimizer_unfolding, optimizer_unet
        )

        metrics = {
            "train_loss":      [],
            "validation_loss": [],
            "train_loss_seg":  [],
            "val_loss_seg":    [],
            "val_loss_total":  [],
            "w_seg":           [],
            "phase":           [],
        }

        for epoch in range(nb_epoch):

            phase         = self._get_phase(epoch, phase1_epochs, phase2_epochs)
            w_seg_current = self._get_w_seg_curriculum(
                epoch, phase1_epochs, phase2_epochs, w_seg_max
            )

            # --- Transition de phase ---
            if phase != current_phase:
                current_phase        = phase
                best_validation_loss = float("inf")
                epoch_no_improve     = 0
                val_history.clear()

                print(f"\n{'='*50}")
                print(f"[PHASE {phase}] Debut a l epoque {epoch+1}")

                if phase == 2:
                    # Reset LR UNet pour phase 2 (repart proprement)
                    self._reset_lr(optimizer_unet, lr_unet_p2)
                    print(f"  LR UNet reset a {lr_unet_p2:.1e}")

                elif phase == 3:
                    # Reset LR des deux modeles pour le fine-tuning
                    self._reset_lr(optimizer_unfolding, lr_unfolding_p3)
                    self._reset_lr(optimizer_unet, lr_unet_p3)
                    print(f"  LR Unfolding reset a {lr_unfolding_p3:.1e}")
                    print(f"  LR UNet reset a {lr_unet_p3:.1e}")

                # Nouveau scheduler par phase
                scheduler_unfolding, scheduler_unet = self._make_schedulers(
                    optimizer_unfolding, optimizer_unet
                )
                print(f"{'='*50}")

            print(f"\n--- Epoque {epoch+1}/{nb_epoch} [Phase {phase} | w_seg={w_seg_current:.3f}] ---")

            avg_train_loss, avg_train_seg = self.train_epoch(
                optimizer_unfolding, optimizer_unet, criterion,
                train_loader, phase, w_seg_current
            )
            avg_validation_loss, avg_val_seg = self.validation_epoch(
                criterion, validation_loader, phase, w_seg_current
            )

            # Metrique de suivi dependante de la phase
            if phase == 1:
                monitor = avg_validation_loss
            elif phase == 2:
                monitor = avg_val_seg
            else:
                monitor = self.w_sr * avg_validation_loss + w_seg_current * avg_val_seg

            avg_val_total = self.w_sr * avg_validation_loss + w_seg_current * avg_val_seg

            val_history.append(monitor)
            val_smoothed = sum(val_history) / len(val_history)

            scheduler_unet.step(val_smoothed)
            scheduler_unfolding.step(val_smoothed)

            metrics["train_loss"].append(avg_train_loss)
            metrics["validation_loss"].append(avg_validation_loss)
            metrics["train_loss_seg"].append(avg_train_seg)
            metrics["val_loss_seg"].append(avg_val_seg)
            metrics["val_loss_total"].append(avg_val_total)
            metrics["w_seg"].append(w_seg_current)
            metrics["phase"].append(phase)

            print(f"  Train loss      : {avg_train_loss:.6f}  (seg : {avg_train_seg:.6f})")
            print(f"  Validation loss : {avg_validation_loss:.6f}  (seg : {avg_val_seg:.6f})")
            print(f"  Monitor         : {monitor:.6f}  (lissee : {val_smoothed:.6f})")
            print(f"  LR Unfolding    : {optimizer_unfolding.param_groups[0]['lr']:.2e} | "
                  f"LR UNet : {optimizer_unet.param_groups[0]['lr']:.2e}")

            best_validation_loss, epoch_no_improve = self.early_stop(
                best_validation_loss, val_smoothed, epoch_no_improve, min_delta
            )

            # Early stop uniquement en phase 3
            if phase == 3 and epoch_no_improve == patience:
                print(f"[EARLY STOP] Arret a l epoque {epoch+1} "
                      f"sans amelioration depuis {patience} epoques.")
                break

            if epoch_no_improve == 0:
                best_model_state = {
                    "unfolding_state":           self.unfolding.state_dict(),
                    "unet_state":                self.unet.state_dict(),
                    "optimizer_unfolding_state": optimizer_unfolding.state_dict(),
                    "optimizer_unet_state":      optimizer_unet.state_dict(),
                    "epoch":                     epoch + 1,
                    "phase":                     phase,
                }
                epoch_save = epoch + 1

        total_duration = time.time() - start_time_total

        if best_model_state is not None:
            torch.save(
                best_model_state,
                os.path.join(output_dir, "best_model.pth")
            )

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n[FIN] Entrainement termine en {total_duration:.2f}s "
              f"({total_duration/60:.2f} min / {total_duration/3600:.2f} h).")
        print(f"[SAVE] Meilleur modele a l epoque {epoch_save} "
              f"dans {os.path.join(output_dir, 'best_model.pth')}")

        plot_metrics(metrics, output_dir)
        return metrics


    def save(self, output_dir, tag="checkpoint"):
        path = os.path.join(output_dir, f"pipe_{tag}.pth")
        torch.save({
            "unfolding_state": self.unfolding.state_dict(),
            "unet_state":      self.unet.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.unfolding.load_state_dict(ckpt["unfolding_state"])
        self.unet.load_state_dict(ckpt["unet_state"])
        print(f"[PipeTrainer] Charge depuis {path}")