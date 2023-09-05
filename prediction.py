import rdkit
from graph_constructor import *
from utils import *
from model_v2 import ITN_V2, Encoder_, DTIConvGraph3Layer, FC, EdgeWeightAndSum_V2
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings
import torch
from dgl.data.utils import Subset

torch.set_default_tensor_type('torch.FloatTensor')
import pandas as pd

warnings.filterwarnings('ignore')
import argparse
import datetime


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bgl, bgp, bglp, y, key_, info_ = batch
        bgl, bgp, bglp, y = bgl.to(device), bgp.to(device), bglp.to(device), y.to(device)
        outputs, _, _ = model(bgl, bgp, bglp)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        #
        # loss.cpu()
        # outputs.cpu()
        # y.cpu()
        # torch.cuda.empty_cache()


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    key = []
    self_attns = []
    weights = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            bgl, bgp, bglp, y, key_, info_ = batch
            bgl, bgp, bglp, y = bgl.to(device), bgp.to(device), bglp.to(device), y.to(device)
            outputs, gp_residue_self_attn, edge_weights = model(bgl, bgp, bglp)
            true.append(y.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key.append(key_)
            # self_attns.append(gp_residue_self_attn)
            # weights.append(edge_weights.data.cpu().numpy())
            # torch.cuda.empty_cache()
    return true, pred, key, self_attns, weights


def collate_fn_v2(data_batch):
    '''
    for the transformer model implemented by torch
    :param data_batch:
    :return:
    '''
    gl_, gp_, glp_, y_, key_, info_ = map(list, zip(*data_batch))
    bgl = dgl.batch(gl_)
    bgp = dgl.batch(gp_)
    bglp = dgl.batch(glp_)
    y = torch.unsqueeze(torch.stack(y_, dim=0), dim=-1)
    return bgl, bgp, bglp, y, key_, info_


path_marker = '/'
data_path = '/home/jingxuan/MHC/baseline/ITN/316'
home_path = '%s/result' %data_path
os.system('mkdir -p %s' %home_path)
os.system('mkdir -p %s/model_save' %home_path)
os.system('mkdir -p %s/stats' %home_path)


class GraphDatasetITN(object):
    def __init__(self, keys=None, labels=None, data_dirs=None, graph_ls_file=None, graph_dic_path=None,
                 pocket_file_path=None, num_process=6, path_marker='/',
                 EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, del_tmp_files=False, distance=800000,
                 max_pocket_num_residues=81 + 1, max_ligand_num_residues=15 + 1):
        self.origin_keys = keys
        self.origin_labels = labels
        self.origin_data_dirs = data_dirs
        self.graph_ls_file = graph_ls_file
        self.graph_dic_path = graph_dic_path
        self.pocket_file_path = pocket_file_path
        self.num_process = num_process
        self.path_marker = path_marker
        self.EtaR = EtaR
        self.ShfR = ShfR
        self.Zeta = Zeta
        self.ShtZ = ShtZ
        self.del_tmp_files = del_tmp_files
        self.distance = distance
        self.max_pocket_num_residues = max_pocket_num_residues
        self.max_ligand_num_residues = max_ligand_num_residues
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.graph_ls_file):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(self.graph_ls_file, 'rb') as f:
                data = pickle.load(f)
            self.gl = data['gl']
            self.gp = data['gp']
            self.glp = data['glp']
            self.keys = data['keys']
            self.labels = data['labels']
            self.infos = data['info']
        else:
            # mk dic path
            if os.path.exists(self.graph_dic_path):
                pass
            else:
                cmdline = 'mkdir -p %s' % (self.graph_dic_path)
                os.system(cmdline)
            pocket_pdb_files = [self.pocket_file_path + self.path_marker + key + '-pro.pdb' for key in self.origin_keys]
            ligand_pdb_files = [self.pocket_file_path + self.path_marker + key + '-pep.pdb' for key in self.origin_keys]
            print('Generate complex graph...')

            # memory friendly
            st = time.time()
            print("main process start >>> pid={}".format(os.getpid()))
            pool = mp.Pool(self.num_process)
            jobs = []
            for i in range(len(self.origin_data_dirs)):
                p = pool.apply_async(partial(graphs_from_mol_itn, EtaR=self.EtaR, ShfR=self.ShfR, Zeta=self.Zeta,
                                             ShtZ=self.ShtZ, max_pocket_num_residues=self.max_pocket_num_residues,
                                             max_ligand_num_residues=self.max_ligand_num_residues,
                                             distance=self.distance),
                                     args=(self.origin_data_dirs[i], self.origin_keys[i], self.origin_labels[i],
                                           pocket_pdb_files[i], ligand_pdb_files[i]))
                jobs.append(p)
                if len(jobs) == 100:
                    write_jobs(jobs, graph_dic_path=self.graph_dic_path, path_marker=self.path_marker)
                    jobs = []
            write_jobs(jobs, graph_dic_path=self.graph_dic_path, path_marker=self.path_marker)
            pool.close()
            pool.join()
            print("main process end (time:%s S)\n" % (time.time() - st))

            # collect the generated graph for each complex
            self.gl = []
            self.gp = []
            self.glp = []
            self.labels = []
            self.infos = []
            self.keys = os.listdir(self.graph_dic_path)
            for key in self.keys:
                with open(self.graph_dic_path + self.path_marker + key, 'rb') as f:
                    graph_dic = pickle.load(f)
                    self.gl.append(graph_dic['gl'])
                    self.gp.append(graph_dic['gp'])
                    self.glp.append(graph_dic['glp'])
                    self.labels.append(graph_dic['label'])
                    self.infos.append(graph_dic['info'])
            # store to the disk
            with open(self.graph_ls_file, 'wb') as f:
                pickle.dump({'gl': self.gl, 'gp': self.gp, 'glp': self.glp, 'keys': self.keys,
                             'labels': self.labels, 'info': self.infos}, f)

            # delete the temporary files
            if self.del_tmp_files:
                cmdline = 'rm -rf %s' % self.graph_dic_path  # graph_dic_path
                os.system(cmdline)

    def __getitem__(self, indx):
        return self.gl[indx], self.gp[indx], self.glp[indx], torch.tensor(eval(self.labels[indx]), dtype=torch.float), \
               self.keys[indx], self.infos[indx]

    def __len__(self):
        return len(self.keys)


class ITN_V2(nn.Module):
    '''
    treat the ligand molecule same as the pocket sequence using transformer (residue sequence)
    '''

    def __init__(self, in_feat_gp=75, d_model=200, d_ff=512, d_k=128, d_v=128, n_heads=4, n_layers=3,
                 dropout=0.20, glp_outdim=200, d_FC_layer=200, n_FC_layer=2, n_tasks=1):
        super(ITN_V2, self).__init__()
        self.d_model = d_model

        # transformer encoder layer for protein pocket sequence
        self.g_trans = Encoder_(in_feat_gp=in_feat_gp, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads,
                                n_layers=n_layers, dropout=dropout)

        # graph layer for ligand and protein residual interaction
        # self.glp_gnn = DTIConvGraph3Layer_V2(in_dim=d_model, out_dim=glp_outdim, dropout=dropout)
        self.glp_gnn = DTIConvGraph3Layer(in_dim=d_model + 1, out_dim=glp_outdim, dropout=dropout)

        # MLP predictor
        self.FC = FC(glp_outdim, d_FC_layer, n_FC_layer, dropout, n_tasks)

        # read out
        self.readout = EdgeWeightAndSum_V2(glp_outdim)

    def forward(self, bgl, bgp, bglp):
        # node representation calculation for the ligand residue sequence
        gl_residue_feats, g1_residue_self_attn = self.g_trans(bgl)  # [batch_size, src_len, d_model]
        gl_residue_feats = gl_residue_feats.view(-1, self.d_model)  # [batch_size*src_len, d_model]

        # node representation calculation for the pocket sequence
        gp_residue_feats, gp_residue_self_attn = self.g_trans(bgp)  # [batch_size, src_len, d_model]
        gp_residue_feats = gp_residue_feats.view(-1, self.d_model)  # [batch_size*src_len, d_model]

        # init the node features of ligand-pocket graph
        bgl_mask = bgl.ndata['pad'].view(-1, 1)  # [batch_size*src_len, 1]
        bgp_mask = bgp.ndata['pad'].view(-1, 1)  # [batch_size*src_len, 1]
        # mask the padding node
        glp_node_feats = gl_residue_feats * bgl_mask + gp_residue_feats * bgp_mask
        bglp_edge_feats = bglp.edata.pop('e')
        # edge update on the ligand-pocket graph
        glp_edge_feats3 = self.glp_gnn(bglp, glp_node_feats, bglp_edge_feats)

        readouts, edge_weights = self.readout(bglp, glp_edge_feats3)
        return torch.sigmoid(self.FC(readouts)), (g1_residue_self_attn, gp_residue_self_attn), edge_weights



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # model training parameters
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -3.5, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=5000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=50, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=0.00, help="L2 regularization")
    argparser.add_argument('--repetitions', type=int, default=5, help="the number of independent runs")
    # transformer parameters
    argparser.add_argument('--n_layers', type=int, default=3, help='the number of encoder layer')
    argparser.add_argument('--in_feat_gp', type=int, default=84)
    argparser.add_argument('--d_ff', type=int, default=512)
    argparser.add_argument('--d_k', type=int, default=128)
    argparser.add_argument('--d_v', type=int, default=128)
    argparser.add_argument('--n_heads', type=int, default=4)
    argparser.add_argument('--d_model', type=int, default=200)
    argparser.add_argument('--dropout', type=float, default=0.25)
    # glp gnn parameters
    argparser.add_argument('--glp_outdim', type=int, default=200, help='the output dim of inter-molecular layers')
    # MLP predictor parameters
    argparser.add_argument('--d_FC_layer', type=int, default=200, help='the hidden layer size of task networks')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--n_tasks', type=int, default=1)
    # others
    argparser.add_argument('--num_workers', type=int, default=0,
                           help='number of workers for loading data in Dataloader')
    argparser.add_argument('--num_process', type=int, default=6,
                           help='number of process for generating graphs')
    argparser.add_argument('--dic_path_suffix', type=str, default='1')
    argparser.add_argument('--distance', type=float, default=80000, help='the distance threshold for '
                                                                       'determining the residue-residue interactions')
    argparser.add_argument('--max_ligand_num_residues', type=int, default=10,
                           help='used for padding, the maximum number of '
                                'ligand residue in the dataset')
    argparser.add_argument('--max_pocket_num_residues', type=int, default=181,
                           help='used for padding, the maximum number of '
                                'pocket residues in the dataset')
    argparser.add_argument('--test_scripts', type=int, default=0,
                           help='whether to test the scripts can run successfully '
                                'using part of datasets (1 for True, 0 for False)')

    args = argparser.parse_args()

    print(args)
    # model training parameters
    gpuid, lr, epochs, batch_size, tolerance, patience, l2, repetitions = args.gpuid, args.lr, args.epochs, args.batch_size, args.tolerance, args.patience, \
                                                                          args.l2, args.repetitions
    # gp transformer parameters
    in_feat_gp, d_ff, d_k, d_v, n_heads, n_layers, d_model, dropout = args.in_feat_gp, args.d_ff, args.d_k, args.d_v, args.n_heads, args.n_layers, \
                                                                      args.d_model, args.dropout
    # glp gnn parameters
    glp_outdim = args.glp_outdim
    # MLP predictor parameters
    d_FC_layer, n_FC_layer, n_tasks = args.d_FC_layer, args.n_FC_layer, args.n_tasks
    # others
    num_workers, num_process, dic_path_suffix, distance, max_ligand_num_residues, max_pocket_num_residues, test_scripts= args.num_workers, args.num_process, \
                                                                                args.dic_path_suffix, args.distance, args.max_ligand_num_residues, \
                                                                                args.max_pocket_num_residues, args.test_scripts

    if test_scripts == 1:
        epochs = 5
        repetitions = 2
        batch_size = 8
        limit = 20
    else:
        limit = None
    

    train_dataset = GraphDatasetITN(keys=None, labels=None, data_dirs=None, graph_ls_file=data_path + path_marker + 'train' + path_marker + 'data.bin',
                                    graph_dic_path=None,pocket_file_path=None, num_process=args.num_process, path_marker='/',
                                    EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, del_tmp_files=False, distance=800000,
                                    max_pocket_num_residues=180 + 1, max_ligand_num_residues=9 + 1)
    valid_dataset = GraphDatasetITN(keys=None, labels=None, data_dirs=None, graph_ls_file=data_path + path_marker + 'valid' + path_marker + 'data.bin',
                                    graph_dic_path=None,pocket_file_path=None, num_process=args.num_process, path_marker='/',
                                    EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, del_tmp_files=False, distance=800000,
                                    max_pocket_num_residues=180 + 1, max_ligand_num_residues=9 + 1)
    test_dataset = GraphDatasetITN(keys=None, labels=None, data_dirs=None, graph_ls_file=data_path + path_marker + 'test' + path_marker + 'data.bin',
                                    graph_dic_path=None,pocket_file_path=None, num_process=args.num_process, path_marker='/',
                                    EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, del_tmp_files=False, distance=800000,
                                    max_pocket_num_residues=180 + 1, max_ligand_num_residues=9 + 1)

    stat_res = []
    print('the number of train data:', len(train_dataset))
    print('the number of valid data:', len(valid_dataset))
    print('the number of test data:', len(test_dataset))
    for repetition_th in range(repetitions):
        torch.cuda.empty_cache()
        dt = datetime.datetime.now()
        filename = home_path + path_marker + 'model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
        print('Independent run %s' % repetition_th)
        print('model file %s' % filename)
        set_random_seed(repetition_th)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=args.num_workers,
                                       collate_fn=collate_fn_v2)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=collate_fn_v2)

        # model
        DTIModel = ITN_V2(in_feat_gp=in_feat_gp, d_model=d_model, d_ff=d_ff, d_k=d_k, d_v=d_v, n_heads=n_heads,
                          n_layers=n_layers,
                          dropout=dropout, glp_outdim=glp_outdim, d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer,
                          n_tasks=n_tasks)
        print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
        if repetition_th == 0:
            print(DTIModel)
        device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        DTIModel.to(device)
        optimizer = torch.optim.Adam(DTIModel.parameters(), lr=lr, weight_decay=l2)

        stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance, filename=filename)
        loss_fn = FocalLoss(gamma=2, alpha=30 / (30 + 1))
        
        train_loss_record = []
        train_auc_record = []
        valid_loss_record = []
        valid_auc_record = []

        for epoch in range(epochs):
            st = time.time()
            # train
            run_a_train_epoch(DTIModel, loss_fn, train_dataloader, optimizer, device)

            # validation
            train_true, train_pred, _, _, _ = run_a_eval_epoch(DTIModel, train_dataloader, device)
            valid_true, valid_pred, _, _, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device)

            train_true = np.concatenate(np.array(train_true), 0)
            train_pred = np.concatenate(np.array(train_pred), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred = np.concatenate(np.array(valid_pred), 0)

            train_loss = loss_fn(torch.tensor(train_pred, dtype=torch.float),
                                 torch.tensor(train_true, dtype=torch.float))
            valid_loss = loss_fn(torch.tensor(valid_pred, dtype=torch.float),
                                 torch.tensor(valid_true, dtype=torch.float))
            train_auc = roc_auc_score(train_true, train_pred)
            valid_auc = roc_auc_score(valid_true, valid_pred)
            
            train_loss_record.append(train_loss)
            train_auc_record.append(train_auc)
            valid_loss_record.append(valid_loss)
            valid_auc_record.append(valid_auc)
            
            early_stop = stopper.step(valid_loss, DTIModel)

            end = time.time()

            if early_stop:
                break
            print("epoch:%s \t train_loss:%.4f \t valid_loss:%.4f \t time:%.3f s" % (
                epoch, train_loss, valid_loss, end - st))
            print("epoch:%s \t train_auc:%.4f \t valid_auc:%.4f \t time:%.3f s" % (
                epoch, train_auc, valid_auc, end - st))

        # load the best model
        stopper.load_checkpoint(DTIModel)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=collate_fn_v2)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=args.num_workers,
                                       collate_fn=collate_fn_v2)
        test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=args.num_workers,
                                      collate_fn=collate_fn_v2)
        train_true, train_pred, tr_keys, train_self_attns, train_weights = run_a_eval_epoch(DTIModel, train_dataloader,
                                                                                            device)
        valid_true, valid_pred, val_keys, valid_self_attns, valid_weights = run_a_eval_epoch(DTIModel, valid_dataloader,
                                                                                             device)
        test_true, test_pred, te_keys, test_self_attns, test_weights = run_a_eval_epoch(DTIModel, test_dataloader,
                                                                                        device)

        # with open(home_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}_weights.bin'.format(
        #         dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), 'wb') as f:
        #     pickle.dump({'train_self_attns': train_self_attns, 'train_weights': train_weights,
        #                  'valid_self_attns': valid_self_attns, 'valid_weights': valid_weights,
        #                  'test_self_attns': test_self_attns, 'test_weights': test_weights, }, f)

        # metrics
        train_true = np.concatenate(np.array(train_true), 0).flatten()
        train_pred = np.concatenate(np.array(train_pred), 0).flatten()
        tr_keys = np.concatenate(np.array(tr_keys), 0).flatten()

        valid_true = np.concatenate(np.array(valid_true), 0).flatten()
        valid_pred = np.concatenate(np.array(valid_pred), 0).flatten()
        val_keys = np.concatenate(np.array(val_keys), 0).flatten()

        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()
        te_keys = np.concatenate(np.array(te_keys), 0).flatten()

        pd_tr = pd.DataFrame({'key': tr_keys, 'train_true': train_true, 'train_pred': train_pred})
        pd_va = pd.DataFrame({'key': val_keys, 'valid_true': valid_true, 'valid_pred': valid_pred})
        pd_te = pd.DataFrame({'key': te_keys, 'test_true': test_true, 'test_pred': test_pred})

        pd_tr.to_csv(
            home_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}_tr.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_va.to_csv(
            home_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}_va.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_te.to_csv(
            home_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}_te.csv'.format(
                dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)

        valid_loss = loss_fn(torch.tensor(valid_pred, dtype=torch.float),
                             torch.tensor(valid_true, dtype=torch.float))
        test_loss = loss_fn(torch.tensor(test_pred, dtype=torch.float), torch.tensor(test_true, dtype=torch.float))
        valid_auc = roc_auc_score(valid_true, valid_pred)
        test_auc = roc_auc_score(test_true, test_pred)

        print('***best ITN model***')
        print("epoch:%s \t valid_loss:%.4f \t test_loss:%.4f" % (
            epoch, valid_loss, test_loss))
        print("epoch:%s \t valid_auc:%.4f \t test_auc:%.4f" % (
            epoch, valid_auc, test_auc))
        stat_res.append([repetition_th, 'train', train_loss, train_auc])
        stat_res.append([repetition_th, 'valid', valid_loss, valid_auc])
        stat_res.append([repetition_th, 'test', test_loss, test_auc])
        
        with open('%s/stats/train_loss_%s.txt'%(home_path,repetition_th),'w')as fo_tr_loss:
            fo_tr_loss.write(str(train_loss_record))
        with open('%s/stats/train_auc_%s.txt'%(home_path,repetition_th),'w')as fo_tr_auc:
            fo_tr_auc.write(str(train_auc_record))
        with open('%s/stats/valid_loss_%s.txt'%(home_path,repetition_th),'w')as fo_va_loss:
            fo_va_loss.write(str(valid_loss_record))
        with open('%s/stats/valid_auc_%s.txt'%(home_path,repetition_th),'w')as fo_va_auc:
            fo_va_auc.write(str(valid_auc_record))
    
    stat_res_pd = pd.DataFrame(stat_res, columns=['repetition', 'group', 'loss', 'auc'])
    stat_res_pd.to_csv(
        home_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
    print(stat_res_pd[stat_res_pd.group == 'train'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'train'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'valid'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'valid'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'test'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'test'].std().values[-4:])
    

    
