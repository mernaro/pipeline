import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.Unet import UNet
from src.utils.UtilsPlot import plot_metrics


class PipeTrainer:

    def __init__(
        self,
        unfolding_model,
        unet_dropout = 0.1,
        w_sr         = 1.0,
        w_seg        = 1.0,
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
        print(f"[PipeTrainer] UNet params : {n_params:,} (Keras reference : 13,423,361)")


    def _seg_loss(self, seg_pred, seg_gt):
        bce  = F.binary_cross_entropy(seg_pred, seg_gt)
        num  = 2.0 * (seg_pred * seg_gt).sum() + 1e-6
        den  = seg_pred.sum() + seg_gt.sum() + 1e-6
        dice = 1.0 - num / den
        return bce + dice


    def train_epoch(self, optimizer, criterion, train_loader, batch_size):
        avg_train_loss = 0.0
        avg_seg_loss   = 0.0
        nb_ite         = 0
        self.unfolding.train()
        self.unet.train()

        for _, (O, L, P, S) in enumerate(train_loader):
            optimizer.zero_grad()
            nb_ite += len(O)

            for i in range(len(O)):
                original_true  = O[i].to(self.device)
                low_resolution = L[i].to(self.device)
                seg_gt         = S[i].to(self.device)

                res_size  = original_true.size()
                inp_size  = low_resolution.size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]

                original_pred = self.unfolding(low_resolution, decim_row, decim_col)

                loss_sr = criterion(original_pred, original_true)

                unet_input = original_pred.squeeze(0).unsqueeze(0).unsqueeze(0)
                seg_pred   = self.unet(unet_input)
                seg_gt_4d  = seg_gt.float().unsqueeze(0).unsqueeze(0)
                loss_seg   = self._seg_loss(seg_pred, seg_gt_4d)

                loss = self.w_sr * loss_sr + self.w_seg * loss_seg
                loss.backward()

                avg_train_loss += loss_sr.item()
                avg_seg_loss   += loss_seg.item()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(self.unfolding.parameters()) + list(self.unet.parameters()),
                    max_norm=self.grad_clip,
                )
            optimizer.step()

        avg_train_loss /= nb_ite
        avg_seg_loss   /= nb_ite
        return avg_train_loss, avg_seg_loss


    def validation_epoch(self, optimizer, criterion, validation_loader, batch_size):
        avg_validation_loss = 0.0
        avg_seg_loss        = 0.0
        nb_ite              = 0
        self.unfolding.eval()
        self.unet.eval()

        with torch.no_grad():
            for _, (O, L, P, S) in enumerate(validation_loader):
                nb_ite += len(O)

                for i in range(len(O)):
                    original_true  = O[i].to(self.device)
                    low_resolution = L[i].to(self.device)
                    seg_gt         = S[i].to(self.device)

                    res_size  = original_true.size()
                    inp_size  = low_resolution.size()
                    decim_row = res_size[0] // inp_size[0]
                    decim_col = res_size[1] // inp_size[1]

                    original_pred = self.unfolding(low_resolution, decim_row, decim_col)

                    loss_sr = criterion(original_pred, original_true)

                    unet_input = original_pred.squeeze(0).unsqueeze(0).unsqueeze(0)
                    seg_pred   = self.unet(unet_input)
                    seg_gt_4d  = seg_gt.float().unsqueeze(0).unsqueeze(0)
                    loss_seg   = self._seg_loss(seg_pred, seg_gt_4d)

                    avg_validation_loss += loss_sr.item()
                    avg_seg_loss        += loss_seg.item()

        avg_validation_loss /= nb_ite
        avg_seg_loss        /= nb_ite
        return avg_validation_loss, avg_seg_loss


    def early_stop(self, best_validation_loss, avg_validation_loss, epoch_no_improve, min_delta):
        if avg_validation_loss + min_delta < best_validation_loss:
            best_validation_loss = avg_validation_loss
            epoch_no_improve     = 0
        else:
            epoch_no_improve += 1
        return best_validation_loss, epoch_no_improve


    def train(
        self,
        optimizer,
        criterion,
        train_loader,
        batch_size_train,
        validation_loader,
        batch_size_val,
        nb_epoch,
        patience,
        output_dir,
        min_delta,
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        start_time_total     = time.time()
        best_validation_loss = float("inf")
        epoch_no_improve     = 0
        best_model_state     = None
        epoch_save           = 0

        # ReduceLROnPlateau surveille la val seg loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = "min",
            factor   = 0.2,
            patience = 10,
            min_lr   = 1e-6,
            verbose  = True,
        )

        metrics = {
            "train_loss":      [],
            "validation_loss": [],
            "train_loss_seg":  [],
            "val_loss_seg":    [],
        }

        for epoch in range(nb_epoch):
            print(f"\n--- Epoque {epoch + 1}/{nb_epoch} ---")

            avg_train_loss, avg_train_seg = self.train_epoch(
                optimizer, criterion, train_loader, batch_size_train
            )
            avg_validation_loss, avg_val_seg = self.validation_epoch(
                optimizer, criterion, validation_loader, batch_size_val
            )

            # Reduction du LR si val seg loss stagne
            scheduler.step(avg_val_seg)

            metrics["train_loss"].append(avg_train_loss)
            metrics["validation_loss"].append(avg_validation_loss)
            metrics["train_loss_seg"].append(avg_train_seg)
            metrics["val_loss_seg"].append(avg_val_seg)

            if hasattr(self.unfolding, "get_metrics"):
                metrics.update(self.unfolding.get_metrics())

            print(f"  Train loss      : {avg_train_loss:.6f}  (seg : {avg_train_seg:.6f})")
            print(f"  Validation loss : {avg_validation_loss:.6f}  (seg : {avg_val_seg:.6f})")
            if hasattr(self.unfolding, "eta"):
                print(f"  eta             : {self.unfolding.eta.item():.6f}")

            best_validation_loss, epoch_no_improve = self.early_stop(
                best_validation_loss, avg_validation_loss, epoch_no_improve, min_delta
            )

            if epoch_no_improve == 0:
                best_model_state = {
                    "unfolding_state": self.unfolding.state_dict(),
                    "unet_state":      self.unet.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                epoch_save = epoch + 1

            if epoch_no_improve == patience:
                print(f"[EARLY STOP] Arret anticipe a l epoque {epoch + 1} "
                      f"car aucune amelioration depuis {patience} epoques.")
                break

        total_duration = time.time() - start_time_total

        if best_model_state is not None:
            torch.save(
                best_model_state,
                os.path.join(output_dir, "best_model.pth")
            )

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n[FIN] Entrainement termine en {total_duration:.2f}s "
              f"(environ {total_duration/60:.2f} min et environ {total_duration/3600:.2f} h).")
        print(f"[SAVE] Meilleur modele sauvegarde dans "
              f"{os.path.join(output_dir, 'best_model.pth')} "
              f"a l epoque n {epoch_save}.")

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