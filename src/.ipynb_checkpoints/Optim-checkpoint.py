import optuna
import argparse
import torch
import torch.nn as nn
import copy
import os
from torch.utils import data
from torch.optim import Adam, AdamW

from models.Unfolding import Unfolding
from src.PipeTrainer import PipeTrainer
from src.datasets.ImageDataset import ImageDataset, get_batch_with_variable_size_image
from src.utils.UtilsLauncher import json_reader, data_config_reader


# -- Nombre d epoques par essai et nombre d essais --------------------
N_EPOCHS_PER_TRIAL = 20
N_TRIALS           = 30
# ---------------------------------------------------------------------


def build_loaders(config):
    """Construit les DataLoaders a partir du config."""
    train_config = config["train"]
    data_dir, train_instances, validation_instances, _ = data_config_reader(config)

    train_dataset = ImageDataset(train_instances,      "train", data_dir=data_dir)
    val_dataset   = ImageDataset(validation_instances, "val",   data_dir=data_dir)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size = train_config["training_batch_size"],
        collate_fn = get_batch_with_variable_size_image,
        shuffle    = True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size = train_config["validation_batch_size"],
        collate_fn = get_batch_with_variable_size_image,
        shuffle    = False,
    )
    return train_loader, val_loader


def objective(trial, config, train_loader, val_loader):
    """
    Fonction objectif Optuna.
    Chaque appel teste une combinaison d hyperparametres
    et retourne la meilleure validation loss SR obtenue.
    """
    # -- Hyperparametres suggeres par Optuna --------------------------
    lr_unfolding = trial.suggest_float("lr_unfolding", 1e-5, 1e-2, log=True)
    lr_unet      = trial.suggest_float("lr_unet",      1e-5, 1e-3, log=True)
    w_sr         = trial.suggest_float("w_sr",         0.5,  2.0)
    w_seg        = trial.suggest_float("w_seg",         0.01, 0.5)
    nb_iteration = trial.suggest_int(  "nb_iteration",  3,    10)

    print(f"\n[Trial {trial.number}] "
          f"lr_unf={lr_unfolding:.2e} | lr_unet={lr_unet:.2e} | "
          f"w_sr={w_sr:.2f} | w_seg={w_seg:.3f} | nb_iter={nb_iteration}")

    # -- Modifie le config pour ce trial -----------------------------
    trial_config = copy.deepcopy(config)
    trial_config["model"]["params"]["nb_iteration"] = nb_iteration

    # -- Initialisation du modele et du trainer -----------------------
    try:
        unfolding_model = Unfolding.from_config(trial_config)
        trainer = PipeTrainer(
            unfolding_model = unfolding_model,
            unet_dropout    = config["train"].get("unet_dropout", 0.2),
            w_sr            = w_sr,
            w_seg           = w_seg,
            grad_clip       = config["train"].get("grad_clip", 1.0),
        )

        weight_decay = config["train"].get("weight_decay", None)
        param_groups = [
            {"params": trainer.unfolding.parameters(), "lr": lr_unfolding},
            {"params": trainer.unet.parameters(),      "lr": lr_unet},
        ]
        if weight_decay is not None:
            optimizer = AdamW(param_groups, weight_decay=weight_decay)
        else:
            optimizer = Adam(param_groups)

        criterion = nn.MSELoss()

        # -- Boucle d entrainement courte -----------------------------
        best_val_sr = float("inf")

        for epoch in range(N_EPOCHS_PER_TRIAL):
            avg_train_loss, _ = trainer.train_epoch(
                optimizer, criterion, train_loader,
                config["train"]["training_batch_size"]
            )
            avg_val_loss, avg_val_seg = trainer.validation_epoch(
                optimizer, criterion, val_loader,
                config["train"]["validation_batch_size"]
            )

            if avg_val_loss < best_val_sr:
                best_val_sr = avg_val_loss

            # Pruning Optuna : arrete les mauvais essais tot
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                print(f"  [Pruned] a l epoque {epoch + 1}")
                raise optuna.exceptions.TrialPruned()

            print(f"  Epoque {epoch + 1:02d}/{N_EPOCHS_PER_TRIAL} | "
                  f"val_SR={avg_val_loss:.5f} | val_seg={avg_val_seg:.5f}")

        print(f"  => Meilleure val_SR : {best_val_sr:.5f}")
        return best_val_sr

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  [Erreur] Trial {trial.number} : {e}")
        return float("inf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for PipeTrainer")
    parser.add_argument("-c", "--config",
                        default="/projects/memaro/mcodjo/unfolding/pipeconfig.json",
                        help="Chemin vers le fichier de config JSON.")
    parser.add_argument("-o", "--output",
                        default="/projects/memaro/mcodjo/unfolding/optuna",
                        help="Dossier de sortie pour les resultats Optuna.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = json_reader(args.config)

    print("Construction des DataLoaders...")
    train_loader, val_loader = build_loaders(config)
    print("DataLoaders prets.")

    # -- Etude Optuna -------------------------------------------------
    # MedianPruner : arrete un essai si sa performance est dans la moitie
    # inferieure des essais precedents a la meme epoque
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction = "minimize",
        pruner    = pruner,
        sampler   = sampler,
        study_name = "pipe_hparam_search",
    )

    study.optimize(
        lambda trial: objective(trial, config, train_loader, val_loader),
        n_trials  = N_TRIALS,
        timeout   = 7200,   # arret automatique apres 2h
        show_progress_bar = True,
    )

    # -- Resultats ----------------------------------------------------
    best = study.best_trial
    print("\n" + "="*60)
    print("MEILLEURE COMBINAISON TROUVEE")
    print("="*60)
    print(f"  val_loss SR      : {best.value:.6f}")
    print(f"  lr_unfolding     : {best.params['lr_unfolding']:.2e}")
    print(f"  lr_unet          : {best.params['lr_unet']:.2e}")
    print(f"  w_sr             : {best.params['w_sr']:.4f}")
    print(f"  w_seg            : {best.params['w_seg']:.4f}")
    print(f"  nb_iteration     : {best.params['nb_iteration']}")
    print("="*60)

    # Sauvegarde du meilleur config
    import json
    best_config = config.copy()
    best_config["train"]["lr_unfolding"] = best.params["lr_unfolding"]
    best_config["train"]["lr_unet"]      = best.params["lr_unet"]
    best_config["train"]["w_sr"]         = best.params["w_sr"]
    best_config["train"]["w_seg"]        = best.params["w_seg"]
    best_config["model"]["params"]["nb_iteration"] = best.params["nb_iteration"]

    best_config_path = os.path.join(args.output, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"\nMeilleur config sauvegarde dans : {best_config_path}")
    print("Tu peux relancer l entrainement avec :")
    print(f"  python pipelauncher.py -c {best_config_path} -a train")

    # Sauvegarde de tous les essais en CSV
    df = study.trials_dataframe()
    csv_path = os.path.join(args.output, "all_trials.csv")
    df.to_csv(csv_path, index=False)
    print(f"Tous les essais sauvegardes dans : {csv_path}")