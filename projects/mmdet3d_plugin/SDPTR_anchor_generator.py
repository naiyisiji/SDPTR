import torch
from torch import nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.runner.base_module import BaseModule
from torch.nn import functional as F

import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        a = self.layer1(x)
        a = self.relu(a)
        a = self.norm(a)
        a = self.layer2(a)
        return a



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Anchor_generator(BaseModule):
    def __init__(self,
                 sample_pts,
                 num_head=8,
                 hidden_dim=128,
                 num_hybrid_layer=2,
                 **kwargs):
        super(Anchor_generator, self).__init__()
        self.hybrid_seq = nn.ModuleList(
            [Anchor_Hybrid_query(sample_pts=sample_pts,num_head=num_head,hidden_dim=hidden_dim) for _ in range(num_hybrid_layer)]
        )

    def forward(self, 
                query,
                sdmap_feats, 
                anchor_ele,             # bs, ele, 2
                anchor_pts,             # bs, ele, pts, 2
                graph_label,
                bev_feats=None, 
                lidar_feats=None):
        device = bev_feats.device
        batch, ele, pts, dim = anchor_pts.shape
        if bev_feats == None:
            raise ValueError('BEV features should not be None')
        feat_size, batch, hidden_dim = bev_feats.shape
        
        query_pts = query
        query_ele = query.view(batch, ele, pts, -1).mean(2)
        m_attn = torch.ones(batch, feat_size).unsqueeze(1).to(device)

        for each_layer in self.hybrid_seq:
            query_pts, query_ele, m_attn, pts_pos = each_layer(query_pts, query_ele, m_attn,
                                           anchor_pts, anchor_ele, graph_label,
                                           sdmap_feats, bev_feats, lidar_feats)
        return query_pts, query_ele, pts_pos
    

class Anchor_Hybrid_query(BaseModule):
    def __init__(self, 
                 dim=2,
                 hidden_dim=128,
                 sample_pts = 10,
                 num_head = 8,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.dim = dim 
        self.hidden_dim = hidden_dim
        self.sample_pts = sample_pts

        self.pos_emb = nn.Linear(dim, hidden_dim)
        self.delta_anchor_pts = nn.Linear(hidden_dim, sample_pts * dim)
        self.attention_simple2query = nn.Linear(hidden_dim, sample_pts)
        self.attention_softmax = nn.Softmax(-1)

        self.bev_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lidar_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.bev_pts_linear_layer = nn.Linear(hidden_dim, hidden_dim)

        self.sdmap_attn = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_head,batch_first=True)
        self.ele_pos_fuse = nn.Parameter(torch.randn(2,1))
        self.ele_pos_embed = nn.Linear(dim, hidden_dim)
        self.ele_anchor_embed = nn.Linear(hidden_dim, hidden_dim)
        self.sdmap_ele_attn = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_head,batch_first=True)

        self.rpts = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead=num_head, batch_first=True)
        self.rele = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
    def forward(self, 
                query_pts,
                query_ele,
                m_attn,
                anchor_pts,
                anchor_ele,
                graph_label,
                sdmap_feat,
                bev_feats,
                lidar_feats=None):
        device = bev_feats.device
        # setting upstream feats, onboard_feats [n, bs, hidden_dim]
        if bev_feats != None and not(lidar_feats):
            onboard_feats = self.bev_proj(bev_feats)
        if lidar_feats != None and not(bev_feats):
            onboard_feats = self.lidar_proj(lidar_feats)
        if bev_feats != None and lidar_feats != None:
            onboard_feats = self.norm(self.bev_proj(bev_feats)) + self.norm(self.lidar_proj(lidar_feats))
        sdmap_pts_feat = sdmap_feat['x_pt']
        sdmap_ele_feat = sdmap_feat['x_pl']

        # process pts querys
        batch, num_ele, num_pts, _ = anchor_pts.shape
        feat_size, batch, hidden_dim = onboard_feats.shape
        
        pts_pos = self.pos_emb(anchor_pts)
        ele_pos = pts_pos.sum(2)
        pts_pos = pts_pos.reshape(batch, num_ele * num_pts, self.hidden_dim)
        query_pts = query_pts + pts_pos
        query_pts = query_pts.reshape(batch, -1, self.hidden_dim)
        delta_anchor_pts = self.delta_anchor_pts(query_pts).reshape(batch, -1, self.sample_pts, self.dim)
        attention_simple_pts = self.attention_softmax(self.attention_simple2query(query_pts))
        anchor_pts_ = anchor_pts.reshape(batch, num_ele * num_pts, 1, 2)
        new_p = anchor_pts_ + delta_anchor_pts
        grid = new_p
        onboard_feats = onboard_feats.permute(1,2,0).unsqueeze(-1)
        grid = 2.0 * grid / (feat_size  -1) -1
        interpolated_values = F.grid_sample(onboard_feats,
                                            grid.view(batch, num_ele * num_pts * self.sample_pts, 1, self.dim),
                                            mode='bilinear',
                                            align_corners=True)
        interpolated_values = interpolated_values.view(batch, hidden_dim, num_pts * num_ele, self.sample_pts).permute(0,2,3,1)
        transformed_value = self.bev_pts_linear_layer(interpolated_values)
        onboard_feat_pts = torch.sum(attention_simple_pts.unsqueeze(-1) * transformed_value, dim=-2)
        sdmap_feat_pts_query = []
        for a in range(batch):
            if len(graph_label == a) == 0:
                sdmap_feat_pts_query_ = query_pts[a,:,:].unsqueeze(0)
                continue
            sdmap_pts_feat_ = sdmap_pts_feat[graph_label == a,:,:].view(-1, hidden_dim).unsqueeze(0)
            sdmap_feat_pts_query_ = self.sdmap_attn(query_pts[a,:,:].unsqueeze(0), sdmap_pts_feat_)
            sdmap_feat_pts_query.append(sdmap_feat_pts_query_)
        sdmap_feat_pts_query = torch.cat(sdmap_feat_pts_query, dim=0)
        onboard_feat_pts_ = onboard_feat_pts + query_pts + sdmap_feat_pts_query # bs, pts, hidden_dim
        
        # process ele querys
        ele_pos_ = self.ele_pos_embed(anchor_ele)
        ele_pos = torch.matmul(torch.stack((ele_pos_, ele_pos), dim=-1),self.ele_pos_fuse).squeeze(-1)
        query_ele = query_ele + ele_pos
        onboard_pos_embed = self.bev_sinusoidal_position_embedding(batch, feat_size, hidden_dim, device)
        onboard_feats = onboard_feats.squeeze(-1).permute(0,2,1) # bs, size, hidden_dim
        
        onboard_feats_ = onboard_feats + onboard_pos_embed
        ele_attn = torch.matmul(query_ele, onboard_feats_.permute(0,2,1))
        ele_attn = F.softmax(ele_attn, dim=-1)
        onboard_feat_ele = torch.matmul(m_attn * ele_attn, onboard_feats_)
        onboard_feat_ele = onboard_feat_ele + query_ele
        m_attn = (ele_attn > 0.5).float()
        sdmap_feat_ele_query = []
        for a in range(batch):
            if len(graph_label == a) == 0:
                sdmap_feat_ele_query_ = query_ele[a,:,:].unsqueeze(0)
                continue
            sdmap_ele_feat_ = sdmap_ele_feat[graph_label == a,:].unsqueeze(0)
            sdmap_feat_ele_query_ = self.sdmap_ele_attn(query_ele[a,:,:].unsqueeze(0), sdmap_ele_feat_)
            sdmap_feat_ele_query.append(sdmap_feat_ele_query_)
        sdmap_feat_ele_query = torch.cat(sdmap_feat_ele_query, dim=0)
        onboard_feat_ele_ = onboard_feat_ele + sdmap_feat_ele_query
        
        x_pts = self.rpts(onboard_feat_pts_)
        x_ele = self.rele(onboard_feat_ele_)
        query_pts = x_pts + x_ele.unsqueeze(-2).repeat(1,1,num_pts,1).view(batch,-1,hidden_dim)
        query_ele = x_ele + torch.sum(x_pts.view(batch, num_ele, num_pts, hidden_dim), dim=-2)

        return query_pts, query_ele, m_attn, pts_pos

    def bev_sinusoidal_position_embedding(self, batch, size, hidden_dim, device, eps=1e-6):
        pos_encoding = torch.zeros(size, hidden_dim)
        position = torch.arange(0, size, dtype=torch.float32).unsqueeze(1)
        theta = torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        theta = 1.0 / (10000 ** (theta / (hidden_dim + eps)))
        pos_encoding[:, 0::2] = torch.sin(position * theta)
        pos_encoding[:, 1::2] = torch.cos(position * theta)
        pos_encoding = pos_encoding.unsqueeze(0).repeat(batch, 1, 1)
        return pos_encoding.to(device)


