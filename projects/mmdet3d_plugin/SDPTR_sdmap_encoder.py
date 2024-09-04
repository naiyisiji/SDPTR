import torch
from torch import nn 
from mmcv.parallel.data_container import DataContainer
from projects.layers.fourier_embedding import FourierEmbedding
from projects.layers.attention_layer import AttentionLayer
from projects.utils.geometry import wrap_angle, angle_between_2d_vectors

from mmcv.runner.base_module import BaseModule
from mmdet.models import NECKS

@NECKS.register_module()
class SDMap_encoder(BaseModule):
    def __init__(self, 
                 num_layers=2,
                 num_heads=8,
                 head_dim=16,
                 hidden_dim=128,
                 num_freq_bands=64,
                 dropout=0.):
        super(SDMap_encoder, self).__init__()
        input_dim_x_pt = 1
        input_dim_x_pl = 0
        input_dim_r_pt2pl = 3
        input_dim_r_pl2pl = 3
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2pl_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.type_embed = nn.Embedding(1, hidden_dim)

        self.pt2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                           bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                           bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
    def forward(self, 
                sdmap_data: torch.Tensor = None, 
                pl2pl_edge: torch.Tensor = None,
                graph_label: torch.Tensor = None):
        device = sdmap_data.device
        if not isinstance(sdmap_data, torch.Tensor):
            raise ValueError(f'In SDMap_encoder module, the class of input data is not Datacontainer, \
                             is {sdmap_data.__class__.__name__}')
        sdmap = sdmap_data.contiguous()
        pl2pl_edge = pl2pl_edge.contiguous()
        graph_label = graph_label.contiguous()
        if sdmap.shape[1] != 50:
            raise ValueError('Input SDMap size is not valid.')
        if sdmap.shape[-1] == 3: # To Do: process 3d dataset
            raise ValueError('The dim of input sdmap is 3')
        elif sdmap.shape[-1] != 2:
            raise ValueError('The dim of input sdmap is invalid')
        
        #print('sdmap shape', sdmap.shape)
        pl_num, pts_num, dim = sdmap.shape

        pts_pos = sdmap.view(-1, 2)
        pts_ori = torch.atan2(-sdmap.flip(dims=[1])[:,:,1], -sdmap.flip(dims=[1])[:,:,0]).view(-1)
        poly_ori = torch.atan2(sdmap[:,1,1] - sdmap[:,0,1], sdmap[:,1,0] - sdmap[:,0,0]).view(-1)
        poly_pos = sdmap[:, 0, :].view(-1, 2)
        pl_ori_vector = torch.stack([poly_ori.cos(), poly_ori.sin()],dim=-1)
        map_magnitude = torch.norm(sdmap[:, :, :2], p=2, dim=-1)
        edge_pts_2_pl = torch.stack([torch.arange(50 * sdmap.shape[0], dtype=torch.long),
                                    torch.arange(sdmap.shape[0], dtype=torch.long).repeat_interleave(50)], dim=0).to(device)
        x_pt = map_magnitude.view(-1).unsqueeze(-1)
        x_pl = None
        
        pts_type_embed = [self.type_embed(torch.zeros(pts_pos.shape[0]).to(device).long())]
        pl_type_embed = [self.type_embed(torch.zeros(poly_pos.shape[0]).to(device).long())]
        x_pt = self.x_pt_emb(continuous_inputs = x_pt, categorical_embs = pts_type_embed)
        x_pl = self.x_pl_emb(continuous_inputs = x_pl, categorical_embs = pl_type_embed)
        rel_pos_pt2pl = pts_pos[edge_pts_2_pl[0]] - poly_pos[edge_pts_2_pl[1]]
        rel_ori_pt2pl = wrap_angle(pts_ori[edge_pts_2_pl[0]] - poly_ori[edge_pts_2_pl[1]])
        r_pt2pl = torch.stack(
            [torch.norm(rel_pos_pt2pl, p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=poly_ori[edge_pts_2_pl[1]],
                                      nbr_vector=rel_pos_pt2pl),
             rel_ori_pt2pl], dim=-1)
        r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None)


        rel_pos_pl2pl = poly_pos[pl2pl_edge[0]] - poly_pos[pl2pl_edge[1]]
        rel_ori_pl2pl = wrap_angle(poly_ori[pl2pl_edge[0]] - poly_ori[pl2pl_edge[1]])
        r_pl2pl = torch.stack(
            [torch.norm(rel_pos_pl2pl, p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=pl_ori_vector[pl2pl_edge[1]],
                                      nbr_vector=rel_pos_pl2pl),
             rel_ori_pl2pl], dim=-1)
        r_pl2pl = self.r_pl2pl_emb(continuous_inputs = r_pl2pl, categorical_embs = None)
        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_pts_2_pl)
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, pl2pl_edge)
        
        x_pl = x_pl.reshape(-1, self.hidden_dim)
        x_pt = x_pt.reshape(-1, pts_num, self.hidden_dim)
        return {'x_pl':x_pl, 'x_pt':x_pt}