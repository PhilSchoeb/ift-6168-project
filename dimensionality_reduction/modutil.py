import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        Dataset personnalisé pour les modèles PyTorch. 

        Paramètres
            X: matrice des caractères (n, d)
            y: vecteur des étiquettes (n,)
        """
        assert isinstance(X,torch.Tensor), f"X doit être un tensor, pas {type(X)}"
        assert isinstance(y,torch.Tensor), f"y doit être un tensor, pas {type(y)}"
        self.X = X
        self.y = y
        self.n_samples = y.shape[0]
        assert self.n_samples==X.shape[0], "X et y ne contiennent pas le même nombre de données"

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def reset_logs():
    logs = {}
    logs["train_times"] = []
    logs["train_losses"] = []
    logs["train_losses_epoch"] = []
    logs["train_scores"] = []
    logs["train_scores_epoch"] = []
    logs["valid_times"] = []
    logs["valid_losses"] = []
    logs["valid_losses_epoch"] = []
    logs["valid_scores"] = []
    logs["valid_scores_epoch"] = []
    logs["best_valid_score"] = -float("inf")
    logs["best_valid_loss"] = float("inf")
    return logs

def evaluate(modele: Module, donnees: DataLoader, fct_perte: Module, fct_score: Callable, device: str = "cpu") -> tuple[float,float,float]:
    modele.eval()
    perte = 0.0
    score = 0.0
    debut_eval = time.perf_counter()
    with torch.no_grad():
        for X, y in donnees:
            logits = modele(X.to(device))
            perte += float(fct_perte(logits,y.to(device)))
            if fct_score:
                score += float(fct_score(logits,y.to(device)))
            else:
                score = 0.0
    return perte/len(donnees), score/len(donnees), time.perf_counter()-debut_eval

def train_one_epoch(modele: Module, optimiseur: Optimizer, fct_perte: Module, fct_score: Callable, train_dataloader: DataLoader, valid_dataloader: DataLoader, epoch: int, logs: dict, device: str = "cpu", strategie_checkpoint: str | int = None, chemin_checkpoints: str = "./checkpoints", verbose: int = 0) -> tuple[float,float,float,float]:
    verbosity = len(train_dataloader)+1 if verbose==0 else int(verbose)
    debut = time.perf_counter()
    debut_log = time.perf_counter()

    # Mise a jour des poids
    perte_train = 0.0
    perte_train_epoch = 0.0
    score_train = 0.0
    score_train_epoch = 0.0
    modele.train()
    for step, batch in enumerate(train_dataloader):
        optimiseur.zero_grad()
        pred = modele(batch[0].to(device))
        perte = fct_perte(pred,batch[1].to(device))
        perte.backward()
        optimiseur.step()
        perte_train += perte.item()
        perte_train_epoch += perte.item()
        if fct_score:
            score = fct_score(pred,batch[1].to(device))
            score_train += score
            score_train_epoch += score
        else:
            score_train = 0.0
            score_train_epoch = 0.0

        # Log des performances
        if (step+1)%verbosity==0 or step+1==len(train_dataloader):
            logs["train_times"].append(time.perf_counter()-debut_log)
            if (step+1)%verbosity==0:
                logs["train_losses"].append(perte_train/verbosity)
                logs["train_scores"].append(score_train/verbosity)
            else:
                logs["train_losses"].append(perte_train/((step+1)%verbosity))
                logs["train_scores"].append(score_train/((step+1)%verbosity))
            if valid_dataloader:
                perte_valid, score_valid, temps_valid = evaluate(modele,valid_dataloader,fct_perte,fct_score,device)
                logs["valid_times"].append(temps_valid)
                logs["valid_losses"].append(perte_valid)
                logs["valid_scores"].append(score_valid)
            
            # Checkpoint
            if strategie_checkpoint=="meilleur" and valid_dataloader:
                if (score_valid>logs["best_valid_score"]) or (score_valid==logs["best_valid_score"] and perte_valid<logs["best_valid_loss"]):
                    logs["best_valid_score"] = score_valid
                    logs["best_valid_loss"] = perte_valid
                    torch.save(modele.state_dict(),f"{chemin_checkpoints}/{modele.__class__.__name__}_best.pth")

        # Checkpoint
        if isinstance(strategie_checkpoint,int):
            if (epoch*len(train_dataloader)+step+1)%strategie_checkpoint==0:
                torch.save(modele.state_dict(),f"{chemin_checkpoints}/{modele.__class__.__name__}_step_{epoch*len(train_dataloader)+step+1}.pth")

        # Affichage des performances
        if (step+1)%verbosity==0:
            temps = (time.perf_counter()-debut)/(step+1)
            restant = int((len(train_dataloader)-step-1)*temps)
            if valid_dataloader:
                print("Itération : {}/{} | Perte train : {:.4f} | Score train : {:.4f} | Perte valid : {:.4f} | Score valid : {:.4f} | ETA : {} m {} s".format(step+1,len(train_dataloader),logs["train_losses"][-1],logs["train_scores"][-1],logs["valid_losses"][-1],logs["valid_scores"][-1],restant//60,restant%60))
            else:
                print("Itération : {}/{} | Perte train : {:.4f} | Score train : {:.4f} | ETA : {} m {} s".format(step+1,len(train_dataloader),logs["train_losses"][-1],logs["train_scores"][-1],restant//60,restant%60))
        if (step+1)%verbosity==0 or step+1==len(train_dataloader):
            debut_log = time.perf_counter()
            perte_train = 0.0
            score_train = 0.0
            modele.train()

    if valid_dataloader:
        return perte_train_epoch/len(train_dataloader), score_train_epoch/len(train_dataloader), perte_valid, score_valid
    return perte_train_epoch/len(train_dataloader), score_train_epoch/len(train_dataloader), None, None

def train(modele: Module, optimiseur: Optimizer, fct_perte: Module, fct_score: Callable, Xy_train: Dataset, Xy_val: Dataset, nb_epochs: int, taille_batch: int, melanger: bool = True, device: str = "cpu", nb_unites: int = 0, chemin_logs: str = "./logs", strategie_checkpoint: str | int = None, chemin_checkpoints: str = "./checkpoints", verbose: int = 0):
    """
    Fonction qui entraine un réseau de neurones. 

    Entrées
        modele: modèle
        optimiseur: optimiseur
        fct_perte: fonction de perte
        fct_score: fonction de score pour évaluer le modèle (None si aucune)
        Xy_train: données d'entrainement
        Xy_val: données de validation (None si aucune)
        nb_epochs: nombre d'epochs à faire
        taille_batch: taille des batchs d'entrainement
        melanger: si on mélange les données
        device: où envoyer les données ('cpu', 'cuda', 'mps', ...)
        nb_unites: nombre d'unités parallèles de calcul à utiliser
        chemin_logs: chemin où enregistrer les logs d'entrainement
        strategie_checkpoint: stratégie d'enregistrement de checkpoints (None, 'meilleur', 'epoch' ou à chaque x itérations)
        chemin_checkpoints: chemin où enregistrer les checkpoints
        verbose: fréquence d'affichage et d'enregistrement de la progression (en itérations)
    """
    assert isinstance(nb_epochs,(int,np.integer,torch.IntType,float,np.floating,torch.FloatType)), f"nb_epochs doit être un int, pas {type(nb_epochs)}"
    assert isinstance(taille_batch,(int,np.integer,torch.IntType,float,np.floating,torch.FloatType)), f"taille_batch doit être un int, pas {type(taille_batch)}"
    assert isinstance(melanger,(bool,np.bool_,torch.BoolType)), f"melanger doit être un bool, pas {type(melanger)}"
    assert isinstance(device,str), f"device doit être un str, pas {type(device)}"
    assert isinstance(nb_unites,(int,np.integer,torch.IntType,float,np.floating,torch.FloatType)), f"nb_unites doit être un int, pas {type(nb_unites)}"
    assert isinstance(chemin_logs,str), f"chemin_logs doit être un str, pas {type(chemin_logs)}"
    assert isinstance(strategie_checkpoint,(type(None),str,int,np.integer,torch.IntType,float,np.floating,torch.FloatType)), f"strategie_checkpoint doit être un str ou un int, pas {type(strategie_checkpoint)}"
    if isinstance(strategie_checkpoint,str):
        assert strategie_checkpoint in ["meilleur","epoch"], "strategie_checkpoint doit être None, 'meilleur', 'epoch', ou un int"
    assert isinstance(chemin_checkpoints,str), f"chemin_checkpoints doit être un str, pas {type(chemin_checkpoints)}"
    assert isinstance(verbose,(int,np.integer,torch.IntType,float,np.floating,torch.FloatType)), f"verbose doit être un int, pas {type(verbose)}"
    
    # Preparer les logs
    os.makedirs(chemin_logs,exist_ok=True)
    logs = reset_logs()
    
    # Preparer les checkpoints
    if strategie_checkpoint:
        os.makedirs(chemin_checkpoints,exist_ok=True)

    # Preparer les donnees
    train_dataloader = DataLoader(Xy_train,batch_size=int(taille_batch),shuffle=melanger,num_workers=int(nb_unites))
    if Xy_val:
        valid_dataloader = DataLoader(Xy_val,batch_size=int(taille_batch),shuffle=melanger,num_workers=int(nb_unites))
    else:
        valid_dataloader = None

    # Entrainement
    debut_train = time.perf_counter()
    for epoch in range(int(nb_epochs)):
        # Mise a jour des poids
        perte_train, score_train, perte_valid, score_valid = train_one_epoch(modele,optimiseur,fct_perte,fct_score,train_dataloader,valid_dataloader,epoch,logs,device,strategie_checkpoint,chemin_checkpoints,verbose)
        logs["train_losses_epoch"].append(perte_train)
        logs["train_scores_epoch"].append(score_train)
        if Xy_val:
            logs["valid_losses_epoch"].append(perte_valid)
            logs["valid_scores_epoch"].append(score_valid)

        # Checkpoint
        if strategie_checkpoint=="epoch":
            torch.save(modele.state_dict(),f"{chemin_checkpoints}/{modele.__class__.__name__}_epoch_{epoch+1}.pth")
        
        # Affichage des performances
        temps = (time.perf_counter()-debut_train)/(epoch+1)
        restant = int((nb_epochs-epoch-1)*temps)
        if Xy_val:
            print("Epoch : {}/{} | Perte train : {:.4f} | Score train : {:.4f} | Perte valid : {:.4f} | Score valid : {:.4f} | ETA : {} m {} s".format(epoch+1,nb_epochs,perte_train,score_train,perte_valid,score_valid,restant//60,restant%60))
        else:
            print("Epoch : {}/{} | Perte train : {:.4f} | Score train : {:.4f} | ETA : {} m {} s".format(epoch+1,nb_epochs,perte_train,score_train,restant//60,restant%60))

    # Affichage des performances finales
    fig, ax = plt.subplots(ncols=2,figsize=(14,5))
    ax[0].plot(range(1,len(logs["train_losses"])+1),logs["train_losses"],color="blue",label="Train")
    ax[1].plot(range(1,len(logs["train_scores"])+1),logs["train_scores"],color="blue",label="Train")
    ax[0].set_title("Perte lors de l'entrainement")
    ax[1].set_title("Score lors de l'entrainement")
    ax[0].set_xlabel("Itérations")
    ax[1].set_xlabel("Itérations")
    ax[0].set_ylabel("Perte")
    ax[1].set_ylabel("Score")
    if Xy_val:
        ax[0].plot(range(1,len(logs["valid_losses"])+1),logs["valid_losses"],color="orange",label="Valid")
        ax[1].plot(range(1,len(logs["valid_scores"])+1),logs["valid_scores"],color="orange",label="Valid")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    plt.show()
    
    # Enregistrement des logs
    f = open(f"{chemin_logs}/resultats_{modele.__class__.__name__}.json","w")
    json.dump(logs,f)
    f.close()
