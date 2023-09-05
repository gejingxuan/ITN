from rdkit import Chem
import dgl
from scipy.spatial import distance_matrix
import torch
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot
from dgl.data.chem import BaseBondFeaturizer
from functools import partial
from itertools import repeat
from torchani import SpeciesConverter, AEVComputer
import multiprocessing as mp
from prody import *
from pylab import *
import pandas as pd
import warnings
import os
import pickle
from dscribe.descriptors import ACSF
from ase import Atoms
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")


# Setting up the ACSF descriptor
acsf = ACSF(
    species=['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'],
    rcut=6.0,
    g2_params=[[4.0, 3.17]],
    g4_params=[[0.1, 3.14, 1]],
)
warnings.filterwarnings('ignore')
converter = SpeciesConverter(['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
AAMAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'ASX': 'B', 'GLX': 'Z', 'SEC': 'U', 'PYL': 'O', 'XLE': 'J'}
Standard_AAMAP = {'HIS': 'H', 'ASP': 'D', 'ARG': 'R', 'PHE': 'F', 'ALA': 'A', 'CYS': 'C', 'GLY': 'G', 'GLN': 'Q',
                  'GLU': 'E', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'SER': 'S', 'TYR': 'Y', 'THR': 'T',
                  'ILE': 'I', 'TRP': 'W', 'PRO': 'P', 'VAL': 'V'}
standd_aa_thr = list(Standard_AAMAP.keys())
standd_aa_one = list(Standard_AAMAP.values())
AABLOSUM62 = pd.read_csv(
    '/home/jingxuan/param/AAD/data/AABLOSUM62.csv')


def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    ab = b - a  
    ac = c - a  
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)

    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()


def ligand_graph(mol, add_self_loop=False):
    g = dgl.DGLGraph()
    num_atoms = mol.GetNumAtoms()  # number of ligand atoms
    g.add_nodes(num_atoms)
    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)
    # add edges, ligand molecule
    num_bonds = mol.GetNumBonds()
    src = []
    dst = []
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
    src_ls = np.concatenate([src, dst])
    dst_ls = np.concatenate([dst, src])
    g.add_edges(src_ls, dst_ls)
    return g, src_ls, dst_ls

def pocket_sequence(pocket):
    """
    :param pocket: pdb object generated using parsePDB function
    :return: sequence for each pocket
    """
    # pocket = parsePDB(pocket_file)
    seq = ''
    if pocket:
        for _ in pocket.iterResidues():
            try:
                letter = standd_aa_one[standd_aa_thr.index(_.getResname())]  # standard amino acids
            except:
                letter = 'U'  # non-standard amino acids
            seq = seq + letter
    else:
        seq = None
    return seq

def construct_complete_graph_from_sequence(sequence, add_self_loop=False):
    """Construct a complete graph using sequence of protein pocket

    The edges are in the order of (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
    If self loops are not created, we will not have (0, 0), (1, 1), ...

    Parameters
    ----------
    sequence : sequence of protein pocket
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty complete graph topology of the protein pocket graph
    """
    num_residues = len(sequence)
    edge_list = []
    for i in range(num_residues):
        for j in range(num_residues):
            if i != j or add_self_loop:
                edge_list.append((i, j))
    g = dgl.DGLGraph(edge_list)

    return g

def construct_graph_from_sequence(sequence):
    """Construct a graph using sequence of protein pocket"""

    num_residues = len(sequence)
    g = dgl.DGLGraph()
    g.add_nodes(num_residues)
    return g

def pocket_featurizer(pocket, pocket_atoms_aves):
    """
    :param pocket: pdb object generated using parsePDB function
    :param pocket_aves:
    :return:
    """
    # pocket = parsePDB(pocket_file)

    # aggregate the ave atom environment descriptor, radius of gyration and BLOSUM62 info for each residual
    gyrations = []
    blosum_info = []
    res_aves = []
    res_atom_ls = []
    for residue in pocket.iterResidues():
        # radius of gyration
        gyrations.append(calcGyradius(residue))

        # BLOSUM62 info
        try:
            letter = standd_aa_one[standd_aa_thr.index(residue.getResname())]  # standard amino acids
            blosum_info.append(AABLOSUM62[AABLOSUM62.iloc[:, 0] == letter].iloc[:, 1:].values[0])
        except:
            blosum_info.append(np.zeros(20))  # non-standard amino acids

        res_atom_ls.append(residue.numAtoms())
    indx = np.cumsum(res_atom_ls)
    res_aves.append(torch.mean(pocket_atoms_aves[0:indx[0]], axis=0).numpy())  # the first residue
    for i in range(pocket.numResidues() - 1):
        res_aves.append(torch.mean(pocket_atoms_aves[indx[i]:indx[i + 1]], axis=0).numpy())

    gyrations_th = torch.unsqueeze(torch.tensor(gyrations, dtype=torch.float), dim=1)  # len = 1
    blosum_info_th = torch.tensor(blosum_info, dtype=torch.float)  # len = 20
    # print(res_aves)
    res_aves_th = torch.tensor(res_aves, dtype=torch.float)  # len = 54
    # print(gyrations_th.shape, blosum_info_th.shape, res_aves_th.shape)
    ndata = torch.cat([gyrations_th, blosum_info_th, res_aves_th], dim=-1)
    # print(ndata.shape)
    return {'x': ndata}

def graphs_from_mol_itn(complex_dir, key, label, pocket_pdb_file, ligand_pdb_file,
                        EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, max_pocket_num_residues=120,
                        max_ligand_num_residues=260, distance=8):
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


    AtomicNums = []
    for i in range(num_atoms_m1):
        AtomicNums.append(mol1.GetAtomWithIdx(i).GetAtomicNum())
    for j in range(num_atoms_m2):
        AtomicNums.append(mol2.GetAtomWithIdx(j).GetAtomicNum())
    Corrds = np.concatenate([mol1.GetConformer().GetPositions(), mol2.GetConformer().GetPositions()], axis=0)
    AtomicNums = torch.tensor(AtomicNums, dtype=torch.long)
    Corrds = torch.tensor(Corrds, dtype=torch.float64)
    AtomicNums = torch.unsqueeze(AtomicNums, dim=0)
    Corrds = torch.unsqueeze(Corrds, dim=0)
    res = converter((AtomicNums, Corrds))
    pbsf_computer = AEVComputer(Rcr=12.0, Rca=12.0, EtaR=torch.tensor([EtaR]), ShfR=torch.tensor([ShfR]),
                                EtaA=torch.tensor([3.5]), Zeta=torch.tensor([Zeta]),
                                ShfA=torch.tensor([0]), ShfZ=torch.tensor([ShtZ]), num_species=9)
    outputs = pbsf_computer((res.species, res.coordinates))
    if torch.any(torch.isnan(outputs.aevs[0].float())):
        print(key)
        status = False
    ligand_atoms_aves = outputs.aevs[0][:num_atoms_m1].float()
    pocket_atoms_aves = outputs.aevs[0][-num_atoms_m2:].float()

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
    gp_dummy1.ndata.update({'x': torch.zeros((ligand_num_residues, 75))})
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

def write_jobs(jobs, graph_dic_path, path_marker):
    for job in jobs:
        dic = job.get()
        if dic is not None:
            with open(graph_dic_path + path_marker + dic['key'], 'wb') as f:
                pickle.dump(dic, f)

class GraphDatasetITN(object):
    def __init__(self, keys=None, labels=None, data_dirs=None, graph_ls_file=None, graph_dic_path=None,
                 pocket_file_path=None, num_process=6, path_marker='/',
                 EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14, del_tmp_files=True, distance=8,
                 max_pocket_num_residues=120, max_ligand_num_residues=260):
        """
        :param keys: the keys for the complexs, list
        :param labels: the corresponding labels for the complexs, list
        :param data_dirs:  the corresponding data_dirs for the complexs, list
        :param graph_ls_file: the cache path for the total graphs objects, labels, keys
        :param graph_dic_path: the cache path for the separate graphs objects (dic) for each complex, do not share the same path with graph_ls_path
        :param pocket_file_path: the storage path for the pdb pocket file
        :param num_process: the numer of process used to generate the graph objects
        :param path_marker: '\\' for windows and '/' for linux
        :param EtaR: parameters for the acsf descriptor
        :param ShfR: parameters for the acsf descriptor
        :param Zeta: parameters for the acsf descriptor
        :param ShtZ: parameters for the acsf descriptor
        :param del_tmp_files: delete the files in the graph_dic_path or not
        """
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
            pocket_pdb_files = [self.pocket_file_path + self.path_marker + key + '_pkt.pdb' for key in self.origin_keys]
            ligand_pdb_files = [self.pocket_file_path + self.path_marker + key + '_lig.pdb' for key in self.origin_keys]
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
                if len(jobs) == 10:
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


