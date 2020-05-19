import os
import os.path as osp
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from utils.read import read_mesh
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from tqdm import tqdm
import numpy as np
import openmesh as om


class ADNI2_clsf(InMemoryDataset):
    url = None

    categories = [
        'ad',
        'cn',
    ]

    def __init__(self,
                 root,
                 train=True,
                 split='clsfb',
                 transform=None,
                 pre_transform=None):
        self.split = split
        self.root = root
        if not osp.exists(osp.join(root, 'processed', self.split)):
            os.makedirs(osp.join(root, 'processed', self.split))

        self.data_files = self.gather_paths(self.split)
        super().__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        all_data_files = []
        for key in self.data_file.keys() :
            all_data_files += self.data_file[key]
        return all_data_files

    @property
    def processed_file_names(self):
        return [
            osp.join(self.split, 'training.pt'),
            osp.join(self.split, 'test.pt')
        ]

    def gather_paths(self, split):
        datapaths = dict()
        count = 0
        if split == 'clsfb' :
            datapaths['ad'] = dict()
            datapaths['cn'] = dict()
            datapaths['ad']['train'] = []
            datapaths['ad']['test'] = []
            datapaths['cn']['train'] = []
            datapaths['cn']['test'] = []

            imgids = np.load(osp.join(self.root, 'adni2_clsfb_dict.npy'), allow_pickle=True)
            imgids = imgids.item()
            for dx in imgids.keys() :
                for tt in imgids[dx].keys() :
                    for i in imgids[dx][tt] :
                        datapaths[dx][tt].append(self.root+'/'+i+'-L_Hipp_first.obj')
                        count += 1
        print('total number of dataset: %d'%(count))
        return datapaths

    def process(self):
        print('Processing...')
        fps = self.data_files

        train_data_list, test_data_list = [], []
        for dx in fps :
            for tt in fps[dx] :
                for idx, fp in enumerate(tqdm(fps[dx][tt])) :
                    mesh = om.read_trimesh(fp)
                    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
                    x = torch.tensor(mesh.points().astype('float32'))
                    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
                    edge_index = to_undirected(edge_index)
                    if dx == 'ad' :
                        data = Data(x=x, y=torch.Tensor([1,0]), edge_index=edge_index, face=face)
                    elif dx == 'cn' :
                        data = Data(x=x, y=torch.Tensor([0,1]), edge_index=edge_index, face=face)
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    if tt == 'test' :
                        test_data_list.append(data)
                    elif tt == 'train' :
                        train_data_list.append(data)

        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(test_data_list), self.processed_paths[1])
