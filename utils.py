import torch
import random
import datetime
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch.nn.functional as F
from rdkit import Chem
from prody import *
import pickle
import numpy as np



def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping(object):
    def __init__(self,  mode='higher', patience=15, filename=None, tolerance=0.0):
        if filename is None:
            dt = datetime.datetime.now()
            filename = './save/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)  # 保存网络中的参数, 速度快，占空间少, 以字典格式存储

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])


class Meter(object):
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_precision_recall_score(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred, pos_label=1)
            scores.append(auc(recall, precision))
        return scores

    def roc_auc_score(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.

        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(torch.nn.functional.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.

        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(torch.nn.functional.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores


    def mae(self):
        """Compute MAE for each task.

        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_absolute_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.

        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(r2_score(task_y_true, task_y_pred))
        return scores

    def Rp(self):
        """Compute Rp for each task.

        Returns
        -------
        list of float
            Rp for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(pearsonr(task_y_true, task_y_pred)[0])
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'prc_auc':
            return self.roc_precision_recall_score()
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'rp':
            return self.Rp()


class Meter_direct(object):
    def __init__(self, y_true, y_pred, mask):
        self.mask = mask
        self.y_pred = y_pred
        self.y_true = y_true

    def roc_precision_recall_score(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred, pos_label=1)
            scores.append(auc(recall, precision))
        return scores

    def roc_auc_score(self):
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.

        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(torch.nn.functional.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.

        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(torch.nn.functional.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores


    def mae(self):
        """Compute MAE for each task.

        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_absolute_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.

        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(r2_score(task_y_true, task_y_pred))
        return scores

    def Rp(self):
        """Compute Rp for each task.

        Returns
        -------
        list of float
            Rp for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(pearsonr(task_y_true, task_y_pred)[0])
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'prc_auc':
            return self.roc_precision_recall_score()
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'rp':
            return self.Rp()


class MyLoss(nn.Module):
    def __init__(self, alph):
        super(MyLoss, self).__init__()
        self.alph = alph

    def forward(self, input, target):
        sum_xy = torch.sum(torch.sum(input * target))
        sum_x = torch.sum(torch.sum(input))
        sum_y = torch.sum(torch.sum(target))
        sum_x2 = torch.sum(torch.sum(input * input))
        sum_y2 = torch.sum(torch.sum(target * target))
        n = input.size()[0]
        pcc = (n * sum_xy - sum_x * sum_y) / torch.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        return self.alph*(1-torch.abs(pcc)) + (1-self.alph)*torch.nn.functional.mse_loss(input, target)



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def pocket_truncate(protein_file, ligand_file, pocket_out_file, complex_out_file, distance=5, sanitize=True):
    ligand = Chem.MolFromMolFile(ligand_file, sanitize=sanitize)
    structure = parsePDB(protein_file)
    if ligand and structure:
        protein = structure.select('protein')  # remove water and other useless
        selected = protein.select('same residue as within %s of ligand' % distance, ligand=ligand.GetConformer().GetPositions())
        writePDB(pocket_out_file, selected)  # contain H
        # to prepare inputs for the model
        pocket = Chem.MolFromPDBFile(pocket_out_file, sanitize=sanitize)  # not contain H
        if pocket:
            Chem.MolToPDBFile(pocket, pocket_out_file)  # not contain H
            with open(complex_out_file, 'wb') as f:
                pickle.dump([ligand, pocket], f)
        else:
            print('pocket file read error for %s' % pocket_out_file)
    elif ligand is None and structure is not None:
        print('only ligand file read error for %s' % ligand_file)
    elif structure is None and ligand is not None:
        print('only protein file read error for %s' % protein_file)
    else:
        print('both protein file and ligand file read error for %s' % protein_file)


def expand_selection(protein_file, selection_out_file, chian_id, residue_num, distance=5, sanitize=True):

    structure = parsePDB(protein_file)
    protein = structure.select('protein')
    if len(chian_id) > 1:
        chain_id1, chain_id2 = chian_id.split(',')
        chain = structure.select('chain %s or chain %s' % (chain_id1, chain_id2))
    else:
        chain = structure.select('chain %s' % chian_id)  # contain H
    residue = chain.select('resnum %s' % residue_num)


    env_residues = protein.select('same residue as within %s of ligand' % distance, ligand=residue.getCoords())
    writePDB(selection_out_file, env_residues)  # contain H

    env_residues = Chem.MolFromPDBFile(selection_out_file, sanitize=sanitize)  # not contain H
    # to prepare inputs for the model
    if env_residues:
        Chem.MolToPDBFile(env_residues, selection_out_file)  # not contain H
    else:
        print('env_residues file read error for %s' % selection_out_file)