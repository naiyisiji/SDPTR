import torch
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmcv.parallel.data_container import DataContainer_topograph as DCT

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

from mmdet3d.datasets.pipelines import DefaultFormatBundle
from typing import Any
from mmdet.datasets.builder import PIPELINES    

@PIPELINES.register_module()
class ExtractSDmapCenterlinePts(object):
    """
    process sdmap to DataContainer
    """
    def __init__(self):
        pass
    def __call__(self, results):
        # Format 3D data
        sdmap = results['sdmap']
        edge_index = torch.nonzero(results['edge'],as_tuple=False).t()
        sdmap = [each_sd[0] for each_sd in sdmap]
        sdmap_ =torch.tensor(sdmap).to(torch.float32)
        
        """
        经过p-mapnet处理的sdmap和ectorize的hdmap是Maptr中所使用的GT经逆时针90度旋转所得
        """
        sdmap_rotated = sdmap_[:, :,[1,0]]
        sdmap_rotated[:, :, 0] *= -1

        results['sdmap'] = dict(sdmap=sdmap_rotated,edge=edge_index)
        for key in ['sdmap']:
            if key not in results:
                continue
            results[key] = DCT(results[key],stack=False)

        return results
