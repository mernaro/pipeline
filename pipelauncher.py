from models.Unfolding import Unfolding
from src.PipeTrainer import PipeTrainer
from src.PipEval import evaluation
from src.datasets.ImageDataset import ImageDataset, get_batch_with_variable_size_image
from src.utils.UtilsLauncher import json_reader, data_config_reader, add_dated_folder, json_saver
from torch.utils import data
from torch.optim import Adam, AdamW
import torch
import torch.nn as nn
import argparse


if __name__ == "__main__":
    print("=== Lancement du script principal (Pipeline) ===")

    parser = argparse.ArgumentParser(description="Training / Evaluation Pipeline")
    parser.add_argument("-c", "--config",
                        default="/projects/memaro/mcodjo/unfolding/pipeconfig.json",
                        help="Chemin vers le fichier de config JSON.")
    parser.add_argument("-a", "--action",
                        default="train",
                        help="Action : train | test")
    args = parser.parse_args()

    print("\n-- ARGUMENTS DU PROGRAMME")
    print(f"\t| Config : {args.config}")
    print(f"\t| Action : {args.action}")

    config       = json_reader(args.config)
    train_config = config["train"]
    output_dir   = add_dated_folder(config["output_dir"])

    data_dir, train_instances, validation_instances, evaluation_instances = data_config_reader(config)

    print("Initialisation des datasets...")
    train_dataset      = ImageDataset(train_instances,      "train", data_dir=data_dir)
    val_dataset        = ImageDataset(validation_instances, "val",   data_dir=data_dir)
    evaluation_dataset = ImageDataset(evaluation_instances, "test",  data_dir=data_dir)

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
        shuffle    = True,
    )
    evaluation_loader = data.DataLoader(
        evaluation_dataset,
        batch_size = train_config["validation_batch_size"],
        collate_fn = get_batch_with_variable_size_image,
        shuffle    = False,
    )
    print(f"Datasets initialises : train({len(train_dataset)}), "
          f"validation({len(val_dataset)}), evaluation({len(evaluation_dataset)}).")

    # Lecture des hyperparametres depuis config
    lr_unfolding = train_config.get("lr_unfolding", train_config["learning_rate"])
    lr_unet      = train_config.get("lr_unet",      train_config["learning_rate"])
    weight_decay = train_config.get("weight_decay", None)
    nb_epochs    = train_config["nb_epochs"]
    patience     = train_config["patience"]
    min_delta    = train_config["min_delta"]
    w_sr         = train_config.get("w_sr",        1.0)
    w_seg        = train_config.get("w_seg",        1.0)
    unet_dropout = train_config.get("unet_dropout", 0.1)
    grad_clip    = train_config.get("grad_clip",    1.0)
    batch_train  = train_config["training_batch_size"]
    batch_val    = train_config["validation_batch_size"]

    if args.action == "train":
        print("Initialisation du modele Unfolding...")
        unfolding_model = Unfolding.from_config(config)
        print("Modele Unfolding initialise.")

        print("Initialisation du PipeTrainer...")
        trainer = PipeTrainer(
            unfolding_model = unfolding_model,
            unet_dropout    = unet_dropout,
            w_sr            = w_sr,
            w_seg           = w_seg,
            grad_clip       = grad_clip,
        )
        print("PipeTrainer initialise.")

        # Deux optimiseurs separes
        print("Initialisation des optimiseurs et de la fonction de perte...")
        if weight_decay is None:
            optimizer_unfolding = Adam(trainer.unfolding.parameters(), lr=lr_unfolding)
            optimizer_unet      = Adam(trainer.unet.parameters(),      lr=lr_unet)
        else:
            optimizer_unfolding = AdamW(trainer.unfolding.parameters(), lr=lr_unfolding, weight_decay=weight_decay)
            optimizer_unet      = AdamW(trainer.unet.parameters(),      lr=lr_unet,      weight_decay=weight_decay)
            print("Optimiseurs AdamW avec weight decay.")

        criterion = nn.MSELoss()
        print("Optimiseurs et fonction de perte initialises.")

        print(f"Debut de l entrainement pour {nb_epochs} epoques...")
        trainer.train(
            optimizer_unfolding = optimizer_unfolding,
            optimizer_unet      = optimizer_unet,
            criterion           = criterion,
            train_loader        = train_loader,
            batch_size_train    = batch_train,
            validation_loader   = val_loader,
            batch_size_val      = batch_val,
            nb_epoch            = nb_epochs,
            patience            = patience,
            output_dir          = output_dir,
            min_delta           = min_delta,
            phase1_epochs       = 200,   
            phase2_epochs       = 200, 
            w_seg_max           = 1, 
            lr_unfolding_p1     = 1e-3,
            lr_unet_p2          = 1e-4,
            lr_unfolding_p3     = 1e-4,
            lr_unet_p3          = 1e-5,
        )

        json_saver(output_dir, config)
        print("=== Entrainement termine avec succes ===")

    elif args.action == "test":
        print("Chargement des modeles pour l evaluation...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        unfolding_model = Unfolding.from_config(config)
        trainer = PipeTrainer(
            unfolding_model = unfolding_model,
            unet_dropout    = unet_dropout,
            w_sr            = w_sr,
            w_seg           = w_seg,
        )

        ckpt = torch.load(config["model_dir"], map_location=device)
        trainer.unfolding.load_state_dict(ckpt["unfolding_state"])
        trainer.unet.load_state_dict(ckpt["unet_state"])
        print(f"Modeles charges depuis {config['model_dir']} (device : {device}).")

        print("Debut de l evaluation du pipeline...")
        evaluation(trainer, evaluation_loader, output_dir)
        print("=== Evaluation terminee avec succes ===")