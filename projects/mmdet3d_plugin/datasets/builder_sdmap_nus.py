import random
from functools import partial
import numpy as np
from torch.utils import data
from mmdet.datasets.samplers import GroupSampler
from mmcv.runner import get_dist_info
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler

import torch
from mmcv.parallel import DataContainer
from mmcv.parallel.data_container import DataContainer_topograph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
def collate_fuse_topograph(batch, samples_per_gpu=1): 
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    
    if isinstance(batch[0], DataContainer_topograph):
        #print(batch[0])
        tmp_batch = []
        for each in batch:
            node_feat = each.data['sdmap']
            #print('node is on ',node_feat.device)
            edge_index = each.data['edge']
            graph = Data(x=node_feat, edge_index=edge_index)
            tmp_batch.append(graph)
            
        tmp=next(iter(DataLoader(tmp_batch, batch_size = samples_per_gpu, shuffle = False)))
        node_feat = tmp.x 
        edge_index = tmp.edge_index
        batch_index = tmp.batch
        return DataContainer([node_feat,edge_index,batch_index],stack=False)

    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate_fuse_topograph(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        tmp = {
            key: collate_fuse_topograph([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
        try:
            assert tmp['sdmap'].data.__len__() == 3
        except:
            raise ValueError('input sdmap is not graph data')
        edge_index = tmp['sdmap'].data[1]
        batch_index = tmp['sdmap'].data[2]
        tmp['edge_index'] = DataContainer([edge_index], stack=False)
        tmp['batch_index'] = DataContainer([batch_index], stack=False)
        return tmp
    else:
        return default_collate(batch)

def build_dataloader_fusion(dataset,
                            samples_per_gpu,
                            workers_per_gpu,
                            num_gpus=1,
                            dist=True,
                            shuffle=True,
                            seed=None,
                            shuffler_sampler=None,
                            nonshuffler_sampler=None,
                            **kwargs):
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )

        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate_fuse_topograph, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)
        
    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


