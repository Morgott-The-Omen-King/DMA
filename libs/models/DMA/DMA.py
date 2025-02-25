import math
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers.shape_spec import ShapeSpec
import fvcore.nn.weight_init as weight_init

from libs.models.DMA.transformers import (SelfAttentionLayer, CrossAttentionLayer, 
                                          FFNLayer, PositionEmbeddingSine3D, MLP)
from libs.models.DAN.resnet import resnet50
from libs.models.DMA.msdeformattn import MSDeformAttnPixelDecoder


class DMA(nn.Module):
    """Decoupled Motion-Appearance Network"""
    def __init__(self, 
                 n_way=1,
                 k_shot=1, 
                 num_support_frames=5,
                 num_query_frames=1,
                 num_meta_motion_queries=100,
                 backbone='resnet50',
                 hidden_dim=256,
                 num_q_former_layers=6):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.num_support_frames = num_support_frames
        self.num_query_frames = num_query_frames
        self.backbone = backbone
    
        self.build_backbone()
        # lateral connections for feature pyramid
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        # feature dimensions for ResNet50
        self.in_features = ["res2", "res3", "res4", "res5"]
        self.in_channels = [256, 512, 1024, 2048]
        self.feat_dim = 256
        # 特征维度
        self.hidden_dim = hidden_dim
        
        # Build lateral and output convolutions for FPN
        # Lateral convs reduce channel dimensions to hidden_dim
        # Output convs do 3x3 convolution on the merged features
        for idx, in_channel in enumerate(self.in_channels):
            lateral_conv = nn.Conv2d(in_channel, self.hidden_dim, kernel_size=1)
            output_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
            
            # Initialize weights using Xavier initialization
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
            
        # ========================================================
        # q-former
        self.num_meta_motion_queries = num_meta_motion_queries
        self.num_q_former_layers = num_q_former_layers
        
        # ========================================================
        # 分割头
        # self.decoder = MaskFormerDecoder(self.hidden_dim)
        self.proposal_generator = MaskProposalGenerator(self.hidden_dim)
        
        self.motion_prototype_network = MotionProtoptyeNetwork(self.hidden_dim, self.num_meta_motion_queries, self.num_q_former_layers)
        self.prototype_enhancer = PrototypeEnhancer(self.hidden_dim)
        
        self.motion_aware_decoder = MotionAwareDecoder(self.hidden_dim)
        self.motion_cls_norm = nn.LayerNorm(self.hidden_dim)
        self.motion_cls_head = nn.Linear(self.hidden_dim, 1)
        # ======================================================== 
    
    @property
    def device(self):
        return next(self.parameters()).device

    def build_backbone(self):
        """构建骨干网络"""
        # 加载预训练的ResNet50
        if self.backbone == 'resnet50':
            backbone = resnet50(pretrained=True)
            # 获取不同层的特征
            self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.conv2, backbone.bn2, backbone.relu2,
                                        backbone.conv3, backbone.bn3, backbone.relu3, backbone.maxpool)
            self.layer1 = backbone.layer1  # 1/4
            self.layer2 = backbone.layer2  # 1/8 
            self.layer3 = backbone.layer3  # 1/16
            self.layer4 = backbone.layer4  # 1/32
            
            # 设置各层特征维度
            self.feat_dims = {
                'layer1': 256,   # layer1 output
                'layer2': 512,   # layer2 output  
                'layer3': 1024, # layer3 output
                'layer4': 2048  # layer4 output
            }
            
            self.feat_dim = 256  # 保持主特征维度不变
        elif self.backbone == 'videoswin':
            raise NotImplementedError("VideoSwin backbone is not implemented yet")
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
    def extract_features(self, rgb_input, mask_feats=False):
        """使用骨干网络提取多尺度特征"""
        # 提取多尺度特征
        feat_1_4 = self.layer1(self.layer0(rgb_input))
        feat_1_8 = self.layer2(feat_1_4)
        feat_1_16 = self.layer3(feat_1_8)
        feat_1_32 = self.layer4(feat_1_16)
        
        # 返回多尺度特征字典
        features = {
            'res2': feat_1_4,
            'res3': feat_1_8,
            'res4': feat_1_16,
            'res5': feat_1_32,
        }
        
        if mask_feats:
            # Generate multi-scale features using FPN
            ms_feats = []
            
            # Start from highest level (1/32 scale)
            prev_features = self.lateral_convs[3](features['res5'])
            ms_feats.append(self.output_convs[3](prev_features))
            
            # Generate 1/16 scale features
            p4 = self.lateral_convs[2](features['res4'])
            p4_up = F.interpolate(prev_features, size=features['res4'].shape[-2:], 
                                mode='bilinear', align_corners=False)
            prev_features = p4 + p4_up
            ms_feats.append(self.output_convs[2](prev_features))
            
            # Generate 1/8 scale features
            p3 = self.lateral_convs[1](features['res3'])
            p3_up = F.interpolate(prev_features, size=features['res3'].shape[-2:],
                                mode='bilinear', align_corners=False)
            prev_features = p3 + p3_up
            ms_feats.append(self.output_convs[1](prev_features))
            
            # Generate 1/4 scale features
            p2 = self.lateral_convs[0](features['res2'])
            p2_up = F.interpolate(prev_features, size=features['res2'].shape[-2:],
                                mode='bilinear', align_corners=False)
            prev_features = p2 + p2_up
            ms_feats.append(self.output_convs[0](prev_features))
    
        features['ms_feats'] = ms_feats
        
        return features
    
    def mask_pooling(self, mask_feats, mask):
        # mask_feats: (B, c, h1, w1)
        # mask: (B, h2, w2)
        # output: (B, C)
        B = mask.shape[0]
        
        # Resize mask to match mask_feats spatial dimensions
        if mask.shape[-2:] != mask_feats.shape[-2:]:
            mask = F.interpolate(mask.unsqueeze(1), size=mask_feats.shape[-2:], 
                               mode='bilinear', align_corners=False).squeeze(1)
        
        mask = mask.view(B, -1)
        # 使用mask pooling提取前景特征
        mask_feats = mask_feats.view(B, -1, mask.shape[-1])
        mask_feats = mask_feats * mask.unsqueeze(1)
        mask_feats = mask_feats.sum(dim=-1) / (mask.sum(dim=-1, keepdim=True) + 1e-6)
        
        return mask_feats
        
    def forward(self, query_video, support_video, support_mask, query_mask=None):
        # Shape检查
        B, T, C, H, W = query_video.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert support_video.shape == (B, self.n_way, self.k_shot, self.num_support_frames, 3, H, W)
        assert support_mask.shape == (B, self.n_way, self.k_shot, self.num_support_frames, H, W)
        
        query_features = self.extract_features(query_video.reshape(B*T, C, H, W), mask_feats=True)
        support_features = self.extract_features(support_video.view(-1, C, H, W), mask_feats=True)
        
        N_way_mask = []
        proposal_masks = []
        motion_cls = []
        for N_way in range(self.n_way):
            # 单独处理每一个way
            s_f32, s_f16, s_f8, s_f4 = support_features['ms_feats']
            s_f32 = s_f32.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f32.shape[-2:])
            s_f16 = s_f16.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f16.shape[-2:])
            s_f8 = s_f8.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f8.shape[-2:])
            s_f4 = s_f4.view(B, self.n_way, self.k_shot, self.num_support_frames, -1, *s_f4.shape[-2:])
            
            s_f32 = s_f32[:, N_way].flatten(0, 2)
            s_f16 = s_f16[:, N_way].flatten(0, 2)
            s_f8 = s_f8[:, N_way].flatten(0, 2)
            s_f4 = s_f4[:, N_way].flatten(0, 2)
            
            s_mask = support_mask[:, N_way].flatten(0, 2)
            
            # support prototype
            s_mask_feats = self.mask_pooling(s_f4, s_mask)
            s_motion_p, s_motion_pe = self.motion_prototype_network(s_mask_feats.view(B, self.k_shot, self.num_support_frames, -1), self.k_shot, self.num_support_frames)
            
            q_f32, q_f16, q_f8, q_f4 = query_features['ms_feats']
            query_proposal_mask = self.proposal_generator(q_f32, q_f16, q_f8, s_motion_p, s_motion_pe)  # 需要添加一个motion_aware模块
            
            # motion prototype
            q_mask_feats = self.mask_pooling(q_f4, query_proposal_mask.sigmoid())
            try:
                q_motion_p, q_motion_pe = self.motion_prototype_network(q_mask_feats.view(B, 1, T, -1), 1, T)
            except Exception as e:
                print(e)
                print(q_mask_feats.shape)
                print(q_f4.shape)
                raise e
            
            enhance_q_motion_p, enhance_q_motion_pe, q_motion_cls = self.prototype_enhancer(q_motion_p, q_motion_pe, s_motion_p, s_motion_pe)
            
            # motion aware decoder
            mask = self.motion_aware_decoder(enhance_q_motion_p, 
                                             enhance_q_motion_pe, 
                                             q_f16.view(B, T, -1, *q_f16.shape[-2:]), 
                                             q_f8.view(B, T, -1, *q_f8.shape[-2:]), 
                                             q_f4.view(B, T, -1, *q_f4.shape[-2:]))
            
            q_motion_cls = self.motion_cls_norm(q_motion_cls)
            q_motion_cls = self.motion_cls_head(q_motion_cls)
            
            N_way_mask.append(mask)
            proposal_masks.append(query_proposal_mask.view(B, T, *query_proposal_mask.shape[-2:]))
            motion_cls.append(q_motion_cls)
        mask = torch.stack(N_way_mask, dim=1)
        proposal_masks = torch.stack(proposal_masks, dim=1)
        motion_cls = torch.stack(motion_cls, dim=1)
        
        return mask, proposal_masks, motion_cls


class MaskProposalGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross attention layers
        self.transformer_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False
            ) for _ in range(2)
        ])
        
        # Feature fusion convs
        self.fusion_conv1 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        self.fusion_conv2 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        
        # Final mask prediction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1)
        )
    
    def forward(self, q_f32, q_f16, q_f8, motion_p, motion_pe):
        # q_f32: (B, c, h/32, w/32)
        # q_f16: (B, c, h/16, w/16)
        # q_f8: (B, c, h/8, w/8)
        # motion_p: (N, B, hidden_dim)
        # motion_pe: (N, B, hidden_dim)
        
        B = q_f16.shape[0]
        T = q_f16.shape[0] // motion_p.shape[1]
        
        # Reshape features for cross attention
        f16_flat = q_f16.flatten(-2).permute(2, 0, 1)  # (HW, B, C)
        f8_flat = q_f8.flatten(-2).permute(2, 0, 1)  # (HW, B, C)
        
        motion_p = motion_p.repeat(1, T, 1)
        motion_pe = motion_pe.repeat(1, T, 1)
                
        # Cross attention with motion prototype
        f16_enhanced = self.transformer_cross_attention_layers[0](
            f16_flat, motion_p,
            pos=motion_pe,
            query_pos=None
        )
        
        f8_enhanced = self.transformer_cross_attention_layers[1](
            f8_flat, motion_p,
            pos=motion_pe,
            query_pos=None
        )
        
        # Reshape back to spatial features
        f16_enhanced = f16_enhanced.permute(1, 2, 0).view(B, -1, *q_f16.shape[-2:])  # (B, C, H/16, W/16)
        f8_enhanced = f8_enhanced.permute(1, 2, 0).view(B, -1, *q_f8.shape[-2:])  # (B, C, H/8, W/8)
        
        # Progressive feature fusion
        f16_up = F.interpolate(f16_enhanced, size=q_f8.shape[-2:], mode='bilinear', align_corners=False)
        fused_8 = self.fusion_conv1(torch.cat([f8_enhanced, f16_up], dim=1))  # (B, C, H/8, W/8)
        
        # Predict mask
        mask = self.mask_conv(fused_8)  # (B, 1, H/8, W/8)
        
        return mask.squeeze(1)


class MotionProtoptyeNetwork(nn.Module):
    def __init__(self, hidden_dim, num_meta_motion_queries=1, num_q_former_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_meta_motion_queries = num_meta_motion_queries
        self.num_q_former_layers = num_q_former_layers
        
        # 特征映射层
        self.motion_proj = nn.Linear(hidden_dim, hidden_dim)
        self.appear_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # q-former
        # self.motion_cls = nn.Linear(1, self.num_meta_motion_queries)
        self.meta_motion_queries = nn.Embedding(1 + self.num_meta_motion_queries, self.hidden_dim)
        self.query_pos_embed = nn.Embedding(1 + self.num_meta_motion_queries, self.hidden_dim)
        
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_1 = nn.ModuleList()
        self.transformer_cross_attention_layers_2 = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_q_former_layers):
            self.transformer_cross_attention_layers_1.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers_2.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=hidden_dim*2,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.meta_motion_queries.weight, std=0.02)
        nn.init.normal_(self.query_pos_embed.weight, std=0.02)
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def get_sinusoid_encoding(self, time_pos, d_hid):
        """支持单序列的正弦位置编码表"""
        # 输入是时间位置tensor
        seq_len = time_pos.shape[0]
        # 创建位置索引
        position = time_pos.float().unsqueeze(-1)  # [seq_len, 1]
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_hid, 2, dtype=torch.float, device=self.device) * 
                           (-math.log(10000.0) / d_hid))  # [d_hid/2]
        
        # 初始化编码表
        sinusoid_table = torch.zeros(seq_len, d_hid, device=self.device)
        
        # 计算正弦和余弦编码
        sinusoid_table[..., 0::2] = torch.sin(position * div_term)  # 偶数维度使用正弦
        sinusoid_table[..., 1::2] = torch.cos(position * div_term)  # 奇数维度使用余弦
        
        return sinusoid_table
        
    def forward(self, features, k_shot, num_support_frames):
        # features: (B, K, T, feat_dim)
        B, K, T, feat_dim = features.shape
        
        # 1. 提取motion和appearance特征
        # 计算相邻帧之间的差异得到motion特征
        motion_feats = features[..., 1:, :] - features[..., :-1, :]
        zero_motion = torch.zeros(B, K, 1, feat_dim, device=self.device)
        motion_feats = torch.cat([motion_feats, zero_motion], dim=2)
        
        # 将motion和appearance特征映射到hidden_dim
        motion_feats = self.motion_proj(motion_feats)
        appear_feats = self.appear_proj(features)
        
        # 2. 添加位置编码
        # 时间位置编码
        time_pos = torch.arange(num_support_frames, device=self.device).repeat(k_shot)
        time_pos_embed = self.get_sinusoid_encoding(time_pos, self.hidden_dim)
        
        # shot位置编码
        shot_pos = torch.arange(k_shot, device=self.device).repeat_interleave(num_support_frames)
        shot_pos_embed = self.get_sinusoid_encoding(shot_pos, self.hidden_dim)
        
        # 合并位置编码
        pos_embed = time_pos_embed + shot_pos_embed
        pos_embed = pos_embed.unsqueeze(1)
        pos_embed = pos_embed.repeat(1, B, 1)
        
        # 3. 初始化meta-motion queries
        motion_q = self.meta_motion_queries.weight.unsqueeze(1).repeat(1, B, 1)
        motion_pe = self.query_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)
        
        try:
            motion_feats = motion_feats.view(B, k_shot * num_support_frames, self.hidden_dim).permute(1, 0, 2)
            appear_feats = appear_feats.view(B, k_shot * num_support_frames, self.hidden_dim).permute(1, 0, 2)
        except Exception as e:
            print(e)
            print(motion_feats.shape)
            print(appear_feats.shape)
            raise e
        
        # 4. 通过q-former提取prototype
        for i in range(self.num_q_former_layers):
            motion_p = self.transformer_cross_attention_layers_1[i](motion_q, motion_feats, pos=pos_embed, query_pos=motion_pe)
            motion_p = self.transformer_self_attention_layers[i](motion_p, tgt_mask=None, tgt_key_padding_mask=None, query_pos=motion_pe)
            motion_p = self.transformer_cross_attention_layers_2[i](motion_p, appear_feats, pos=pos_embed, query_pos=motion_pe)
            motion_p = self.transformer_ffn_layers[i](motion_p)
            
        return motion_p, motion_pe  # (num_queries, B, hidden_dim)
        

class PrototypeEnhancer(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=hidden_dim*2,
                    dropout=0.0,
                )
            )
            
    def forward(self, motion1_p, motion1_pe, motion2_p, motion2_pe):
        for i in range(self.num_layers):
            motion1_p = self.transformer_cross_attention_layers[i](motion1_p, motion2_p, pos=motion2_pe, query_pos=motion1_pe)
            motion1_p = self.transformer_self_attention_layers[i](motion1_p, tgt_mask=None, tgt_key_padding_mask=None, query_pos=motion1_pe)
            motion1_p = self.transformer_ffn_layers[i](motion1_p)
        
        return motion1_p[1:], motion1_pe[1:], motion1_p[0]
    
    
class MotionAwareDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross attention layers
        self.transformer_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False
            ) for _ in range(3)
        ])
        
        # Feature fusion convs
        self.fusion_conv1 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        self.fusion_conv2 = nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1)
        
        # Final mask prediction
        self.mask_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, 1, 1)
        )
        
    def forward(self, enhance_q_motion_p, enhance_q_motion_pe, f16, f8, f4):
        # enhance_q_motion_p: (1, B, hidden_dim)
        # enhance_q_motion_pe: (1, B, hidden_dim) 
        # f16: (B, T, hidden_dim, H/16, W/16)
        # f8: (B, T, hidden_dim, H/8, W/8)
        # f4: (B, T, hidden_dim, H/4, W/4)
        
        B, T = f16.shape[0], f16.shape[1]
        
        # Reshape features for cross attention
        f16_flat = f16.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        f8_flat = f8.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        f4_flat = f4.flatten(0, 1).flatten(-2).permute(2, 0, 1)  # (HW, B*T, C)
        
        # Expand motion prototype to match batch size
        enhance_q_motion_p = enhance_q_motion_p.repeat(1, T, 1)  # (1, B*T, hidden_dim)
        enhance_q_motion_pe = enhance_q_motion_pe.repeat(1, T, 1)  # (1, B*T, hidden_dim)
        
        # Cross attention with motion prototype
        f16_enhanced = self.transformer_cross_attention_layers[0](
            f16_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        f8_enhanced = self.transformer_cross_attention_layers[1](
            f8_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        f4_enhanced = self.transformer_cross_attention_layers[2](
            f4_flat, enhance_q_motion_p,
            pos=enhance_q_motion_pe,
            query_pos=None
        )
        
        # Reshape back to spatial features
        f16_enhanced = f16_enhanced.permute(1, 2, 0).view(B, T, -1, *f16.shape[-2:])  # (B, T, C, H/16, W/16)
        f8_enhanced = f8_enhanced.permute(1, 2, 0).view(B, T, -1, *f8.shape[-2:])  # (B, T, C, H/8, W/8)
        f4_enhanced = f4_enhanced.permute(1, 2, 0).view(B, T, -1, *f4.shape[-2:])  # (B, T, C, H/4, W/4)
        
        # Progressive feature fusion
        f16_up = F.interpolate(f16_enhanced.flatten(0,1), size=f8.shape[-2:], mode='bilinear', align_corners=False)
        # f16_up = f16_up.view(B, T, -1, *f8.shape[-2:])  # (B, T, C, H/8, W/8)
        fused_8 = self.fusion_conv1(torch.cat([f8_enhanced.flatten(0,1), f16_up], dim=1))  # (B, T, C, H/8, W/8)
        
        fused_8_up = F.interpolate(fused_8, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        # fused_8_up = fused_8_up.view(B, T, -1, *f4.shape[-2:])  # (B, T, C, H/4, W/4)
        fused_4 = self.fusion_conv2(torch.cat([f4_enhanced.flatten(0,1), fused_8_up], dim=1))  # (B, T, C, H/4, W/4)
        
        # Predict mask
        mask = self.mask_conv(fused_4)  # (B*T, 1, H/4, W/4)
        mask = mask.view(B, T, *mask.shape[-2:])  # (B, T, H/4, W/4)
        
        return mask
        

# 测试代码
if __name__ == "__main__":
    # # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_way = 3
    k_shot = 2
    num_support_frames = 4
    model = DMA(n_way=n_way, k_shot=k_shot, num_support_frames=num_support_frames, num_meta_motion_queries=20).to(device)
    
    # Create test data
    B, T = 2, 16  # batch size, num frames
    H, W = 241, 425  # height, width
    query_video = torch.randn(B, T, 3, H, W).to(device)
    support_video = torch.randn(B, n_way, k_shot, num_support_frames, 3, H, W).to(device)
    support_mask = torch.randn(B, n_way, k_shot, num_support_frames, H, W).to(device)
    
    # Forward pass
    masks = model(query_video, support_video, support_mask)
    print("Output masks shape:", masks.shape)
    
    # pixel_decoder = PixelDecoder()