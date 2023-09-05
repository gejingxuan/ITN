import pickle
import os
from turtle import home
import pandas as pd
import re
from rdkit import Chem
from prody import *
from graph_constructor import *
import argparse
from utils import *
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
# from model_v2 import IGN_New, IGN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dgl.data.utils import Subset
import time
import warnings
from dscribe.descriptors import ACSF
acsf = ACSF(
    species=['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'],
    rcut=12.0,
    g2_params=[[4.0, 3.17]],
    g4_params=[[0.1, 3.14, 1]],
)

warnings.filterwarnings("ignore")


def sep_files(file):
    cmdline = 'tail -n +2 %s | grep -n TER' % file
    res = os.popen(cmdline).read()
    ter_idx = eval(res.split(':')[0])


    cmdline = 'head -n 1 %s' % file
    res = os.popen(cmdline).read()
    res_ls = res.strip().split('\t')
    name = res_ls[0] + '_' + res_ls[1] + '_' + res_ls[2] + '_' + res_ls[3]

    cmdline = 'head -n %s %s > %s/sep/%s-pro.pdb &&' % (ter_idx + 1, file, home_path, name)
    cmdline += 'tail -n +%s %s > %s/sep/%s-pep.pdb' % (ter_idx + 2, file, home_path, name)
    os.system(cmdline)

    pro = Chem.MolFromPDBFile('%s/sep/%s-pro.pdb' % (home_path, name))  # not contain H
    pep = Chem.MolFromPDBFile('%s/sep/%s-pep.pdb' % (home_path, name))  # not contain H
    if pro and pep:
        Chem.MolToPDBFile(pro, '%s/sep/%s-pro.pdb' % (home_path, name))
        Chem.MolToPDBFile(pep, '%s/sep/%s-pep.pdb' % (home_path, name))
        with open('%s/comp'% home_path + path_marker + name, 'wb') as f:
            pickle.dump([pep, pro], f)
    else:
        print('rdk reading error for %s' % name)
        cmdline = 'rm -rf %s/sep/%s-pro.pdb &&' % (home_path, name)
        cmdline += 'rm -rf %s/sep/%s-pep.pdb' % (home_path, name)
        os.system(cmdline)


def graphs_from_mol_itn(complex_dir, key, label, pocket_pdb_file, ligand_pdb_file,
                        EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, max_pocket_num_residues=181,
                        max_ligand_num_residues=10, distance=80000):
    try:
        status = True
        dis_threshold = distance
        # try:
        with open(complex_dir, 'rb') as f:
            mol1, mol2 = pickle.load(f)  # mol1 is the ligand, mol2 is the pocket
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
        # num_atoms = num_atoms_m1 + num_atoms_m2  # total number of atoms
        pocket = parsePDB(pocket_pdb_file)
        ligand = parsePDB(ligand_pdb_file)
        ligand_num_residues = ligand.numResidues()
        pocket_num_residues = pocket.numResidues()

        # DScribe package
        atom_ls = []
        atom_ls.extend([atom.GetSymbol() for atom in mol1.GetAtoms()])
        atom_ls.extend([atom.GetSymbol() for atom in mol2.GetAtoms()])
        atom_positions = np.concatenate([mol1.GetConformer().GetPositions(), mol2.GetConformer().GetPositions()],
                                        axis=0)
        mol_ase = Atoms(symbols=atom_ls, positions=atom_positions)
        res = acsf.create(mol_ase)
        res_th = torch.tensor(res, dtype=torch.float)
        if torch.any(torch.isnan(res_th)):
            print(key)
            status = False
        ligand_atoms_aves = res_th[:num_atoms_m1].float()
        pocket_atoms_aves = res_th[-num_atoms_m2:].float()

        # construct ligand graph
        ligand_seq = pocket_sequence(ligand)
        gl = construct_graph_from_sequence(ligand_seq)
        gl.ndata.update(pocket_featurizer(ligand, ligand_atoms_aves))
        gl.ndata.update({'pad': torch.ones(ligand_num_residues)})
        # add padding node in gl
        gl.add_nodes(max_ligand_num_residues + max_pocket_num_residues - ligand_num_residues)

        # construct pocket graph
        # add padding node part1 in gp
        gp_dummy1 = dgl.DGLGraph()
        gp_dummy1.add_nodes(ligand_num_residues)
        gp_dummy1.ndata.update({'x': torch.zeros((ligand_num_residues, 84))})
        gp_dummy1.ndata.update({'pad': torch.zeros(ligand_num_residues)})

        pocket_seq = pocket_sequence(pocket)
        gp = construct_graph_from_sequence(pocket_seq)
        # featurizer the pcoket graph
        gp.ndata.update(pocket_featurizer(pocket, pocket_atoms_aves))
        gp.ndata.update({'pad': torch.ones(pocket_num_residues)})

        gp = dgl.batch([gp_dummy1, gp])
        gp.flatten()

        # add padding node part2 in gp
        gp.add_nodes(max_ligand_num_residues + max_pocket_num_residues - ligand_num_residues - pocket_num_residues)

        glp = dgl.DGLGraph()
        glp.add_nodes(ligand_num_residues + pocket_num_residues)

        lig_res_geo_center = []
        for res in ligand.iterResidues():
            lig_res_geo_center.append(res.getCoords().mean(axis=0))
        lig_res_geo_center = np.array(lig_res_geo_center)

        pkt_res_geo_center = []
        for res in pocket.iterResidues():
            pkt_res_geo_center.append(res.getCoords().mean(axis=0))
        pkt_res_geo_center = np.array(pkt_res_geo_center)

        dis_matrix = distance_matrix(lig_res_geo_center, pkt_res_geo_center)
        node_idx = np.where(dis_matrix < dis_threshold)
        src_ls3 = np.concatenate([node_idx[0]])
        dst_ls3 = np.concatenate([node_idx[1] + ligand_num_residues])
        # add edges
        glp.add_edges(src_ls3, dst_ls3)

        # add distance info for glp edge
        inter_dis = dis_matrix[node_idx[0], node_idx[1]]
        glp_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)
        glp.edata['e'] = glp_d * 0.1

        # add padding node in glp
        glp.add_nodes(max_ligand_num_residues + max_pocket_num_residues - ligand_num_residues - pocket_num_residues)

        if torch.any(torch.isnan(gl.ndata['x'])) or torch.any(torch.isnan(gp.ndata['x'])):
            status = False
            print('nan error', key)
        if status:
            return {'gl': gl, 'gp': gp, 'glp': glp, 'key': key, 'label': label,
                    'info': {'ligand_num_residues': ligand_num_residues, 'pocket_num_residues': pocket_num_residues}}
    except:
        print('error', key)


class GraphDatasetITN(object):
    def __init__(self, keys=None, labels=None, data_dirs=None, graph_ls_file=None, graph_dic_path=None,
                 pocket_file_path=None, num_process=32, path_marker='/',
                 EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, del_tmp_files=False, distance=800000,
                 max_pocket_num_residues=180+1, max_ligand_num_residues=10+1):
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
                                             max_ligand_num_residues=self.max_ligand_num_residues, distance=self.distance),
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
        return self.gl[indx], self.gp[indx], self.glp[indx], torch.tensor(self.labels[indx], dtype=torch.float), \
               self.keys[indx], self.infos[indx]

    def __len__(self):
        return len(self.keys)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--group', type=str, default='316')
    argparser.add_argument('--dataset', type=str, default='train')
    argparser.add_argument('--num_process', type=int, default='16')

    args = argparser.parse_args()

    st = time.time()

    group = args.group
    dataset_name = args.dataset
    home_path = '/home/jingxuan/MHC/baseline/ITN/%s/%s' %(group,dataset_name)
    num_process = args.num_process
    path_marker = '/'

    os.system('mkdir -p %s/comp' %home_path)
    os.system('mkdir -p %s/sep' %home_path)
    for system_index in os.listdir('/home/jingxuan/MHC/split_%s/%s' %(group,dataset_name)):
        sep_files('/home/jingxuan/MHC/split_%s/%s/%s' %(group,dataset_name,system_index))

    total_keys = os.listdir('%s/comp' %home_path)
    total_labels = []
    for total_key in total_keys:
        system, name, seq, label = total_key.split('_')
        total_labels.append(label)
    total_dirs_new = []
    for key in total_keys:
        total_dirs_new.append('%s/comp' %home_path + path_marker + key)

    dataset = GraphDatasetITN(keys=total_keys, labels=total_labels,
                            data_dirs=total_dirs_new,
                            graph_ls_file=home_path + path_marker + 'data.bin',
                            graph_dic_path=home_path + path_marker + 'sep_graphs',
                            num_process=num_process,
                            distance=800000, path_marker='/', pocket_file_path='%s/sep' %home_path)
    end = time.time()
    print(end - st, 'S')
