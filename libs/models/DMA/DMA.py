import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


from libs.models.DMA.transformers import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, PositionEmbeddingSine3D, MLP
from libs.models.DAN.resnet import resnet50
from libs.models.DMA.msdeformattn import MSDeformAttnPixelDecoder

from detectron2.layers.shape_spec import ShapeSpec
import fvcore.nn.weight_init as weight_init


class DMA(nn.Module):
    """Decoupled Motion-Appearance Network"""
    def __init__(self, 
                 n_way=1,
                 k_shot=1, 
                 num_support_frames=5,
                 num_query_frames=1,
                 num_meta_motion_queries=1,
                 hidden_dim=256,
                 backbone='resnet50',
                 num_q_former_layers=6):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.num_support_frames = num_support_frames
        self.num_query_frames = num_query_frames
        self.backbone = backbone
    
        self.build_backbone()
            
        # 隐藏特征维度
        self.hidden_dim = hidden_dim
        
        # ========================================================
        # 特征映射层
        self.motion_proj = nn.Linear(self.feat_dim, self.hidden_dim)
        self.appear_proj = nn.Linear(self.feat_dim, self.hidden_dim)
        # ========================================================
        
        # ========================================================
        # q-former
        self.num_meta_motion_queries = num_meta_motion_queries
        self.meta_motion_queries = nn.Embedding(self.num_meta_motion_queries, self.hidden_dim)
        self.query_pos_embed = nn.Embedding(self.num_meta_motion_queries, self.hidden_dim)
        
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers_1 = nn.ModuleList()
        self.transformer_cross_attention_layers_2 = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.num_q_former_layers = num_q_former_layers
        
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
                    dim_feedforward=self.hidden_dim*2,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
        # ========================================================
        
        # ========================================================
        # 分割头
        self.decoder = MaskFormerDecoder(self.hidden_dim)
        self.mask_head = nn.Conv2d(self.hidden_dim, self.n_way, kernel_size=1)
        self.pixel_decoder = PixelDecoder()
        # ========================================================
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Special initialization for meta motion queries and positional embeddings
        nn.init.normal_(self.meta_motion_queries.weight, std=0.02)
        nn.init.normal_(self.query_pos_embed.weight, std=0.02)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def build_backbone(self):
        """构建骨干网络"""
        # 加载预训练的ResNet50
        self.feat_dim = 256  # 保持主特征维度不变
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
        elif self.backbone == 'resnet50_official':
            from torchvision.models import resnet50_official, ResNet50_Weights
            backbone = resnet50_official(weights=ResNet50_Weights.IMAGENET1K_V2)
            # layer0: 初始卷积 + BN + ReLU + MaxPool
            self.layer0 = nn.Sequential(
                backbone.conv1,  # 7x7 conv, stride=2
                backbone.bn1,    # batch norm
                backbone.relu,   # ReLU
                backbone.maxpool # 3x3 maxpool, stride=2
            )
            self.layer1 = backbone.layer1  # 1/4
            self.layer2 = backbone.layer2  # 1/8 
            self.layer3 = backbone.layer3  # 1/16
            self.layer4 = backbone.layer4  # 1/32
            
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
            # permute 为 B H W C
            mask_feats, _, ms_feats = self.pixel_decoder(features)
        
        features['mask_feats'] = mask_feats
        features['ms_feats'] = ms_feats
        
        return features
    
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
        
    def forward(self, query_video, support_video, support_mask):
        # Shape检查
        B, T, C, H, W = query_video.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert support_video.shape == (B, self.n_way, self.k_shot, self.num_support_frames, 3, H, W)
        assert support_mask.shape == (B, self.n_way, self.k_shot, self.num_support_frames, H, W)
        
        # Step 1: 特征提取
        # Query视频特征
        query_feats_ = self.extract_features(query_video.reshape(B*T, C, H, W), mask_feats=True)
        
        query_mask_feats, query_feats = query_feats_['mask_feats'], query_feats_['mask_feats']
        
        
        feat_h, feat_w = query_feats.shape[-2:]  # 获取特征图的实际高宽
        query_feats = query_feats.view(B, T, self.feat_dim, feat_h, feat_w)
        
        # Support视频特征
        support_feats = self.extract_features(support_video.view(-1, C, H, W), mask_feats=True)
        support_mask_feats, support_feats = support_feats['mask_feats'], support_feats['mask_feats']  # bs*n_way*k_shot*num_support_frames, feat_dim, feat_h, feat_w
        support_feats = support_feats.view(B, self.n_way, self.k_shot, self.num_support_frames, self.feat_dim, feat_h, feat_w)
        # 提取对应的support对应的appearance feat以及motion feat
        # 1. 提取使用mask pooling得到对应前景特帧，这个表示物体的appearance 特征
        # 2. 将两两帧之间的appearance特征进行相减，得到对应的motion 特帧
        
        # 1. 使用mask pooling提取前景特征
        support_mask_resized = F.interpolate(
            support_mask.view(-1, 1, H, W).float(),
            size=(feat_h, feat_w),
            mode='bilinear',
            align_corners=False
        )
        
        support_feats_flat = support_feats.view(-1, self.feat_dim, feat_h, feat_w)
        
        # 计算前景区域的平均值
        masked_feats = support_feats_flat * support_mask_resized
        mask_sum = support_mask_resized.sum(dim=(2,3), keepdim=True) + 1e-6
        masked_feats = (masked_feats.sum(dim=(2,3), keepdim=True) / mask_sum).squeeze(-1).squeeze(-1)
        
        appearance_feats = masked_feats.view(B, self.n_way, self.k_shot, self.num_support_frames, self.feat_dim)  # (B, n_way, k_shot, num_support_frames, feat_dim)
        
        # 2. 计算相邻帧之间的差异得到motion特征
        # 使用切片并行计算相邻帧差异
        # motion_feats = appearance_feats[..., 1:, :] - appearance_feats[..., :-1, :]
        # # 补充一个空的motion特征,保持时间维度一致
        # zero_motion = torch.zeros_like(motion_feats[...,:1,:])
        # motion_feats = torch.cat([motion_feats, zero_motion], dim=3)
        motion_feats = appearance_feats
        
        # 将motion和appearance特征映射到hidden_dim
        motion_feats = self.motion_proj(motion_feats)
        appearance_feats = self.appear_proj(appearance_feats)
        
        ######################################## todo
        # 提取对应的meta-motion prototype
        # 1. 实现一个Q-former，用于提取meta-motion prototype
        # 这个Q-former的输入包括motion_feats和appearance_feats，和模型参数meta_motion_queries
        # 每一层的q-former包含cross-attention(交互到motion_feats)+cross-attention(交互到appearance_feats)+self-attention(meta_motion_queries自己交互)，一个FFN，输出就是对应的meta_motion_prototype
        # 由于我们有n-way，k-shot的support set，那么提取的时候，我们就有对饮的n-way组meta-motion_prototype, 每一组meta_motion_prototype只去学习自己对应的motion_feats和appearance_feats
        # 不同shot的motion_feats和appearance_feats要concat在一起，但是在做attention的时候一定要加上合适的位置编码（时间位置编码以及对应shot的位置编码，让模型知道这是不同的shot）
        # 为每个way单独提取meta-motion prototype
        
        # 我这里在具体一点
        # 输入到Q-former中（q-former层数为6）
        # meta_motion_queries: (num_meta_motion_queries, b*n_way, c)
        # support_motion_feats: (k_shot * num_support, b*n_way, c)
        # support_appearance_feats: (k_shot * num_support, b*n_way, c)
        # 1. 准备输入数据
        # 将motion和appearance特征展平并拼接
        motion_feats = motion_feats.view(B * self.n_way, self.k_shot * self.num_support_frames, self.hidden_dim).permute(1, 0, 2)  # (k_shot*num_support_frames, B*n_way, hidden_dim)
        appearance_feats = appearance_feats.view(B * self.n_way, self.k_shot * self.num_support_frames, self.hidden_dim).permute(1, 0, 2)  # (k_shot*num_support_frames, B*n_way, hidden_dim)
        
        # 2. 添加位置编码
        # 时间位置编码
        time_pos = torch.arange(self.num_support_frames, device=self.device).repeat(self.k_shot)
        time_pos_embed = self.get_sinusoid_encoding(time_pos, self.hidden_dim)
        
        # shot位置编码
        shot_pos = torch.arange(self.k_shot, device=self.device).repeat_interleave(self.num_support_frames)
        shot_pos_embed = self.get_sinusoid_encoding(shot_pos, self.hidden_dim)
        
        # 合并位置编码
        pos_embed = time_pos_embed + shot_pos_embed
        pos_embed = pos_embed.unsqueeze(1)
        pos_embed = pos_embed.repeat(1, B * self.n_way, 1)
        
        # 3. 初始化meta-motion queries
        meta_motion_queries = self.meta_motion_queries.weight.unsqueeze(1).repeat(1, B * self.n_way, 1)
        query_pos_embed = self.query_pos_embed.weight.unsqueeze(1).repeat(1, B * self.n_way, 1)
        
        
        # motion_feats = motion_feats.permute(1, 0, 2)
        # appearance_feats = appearance_feats.permute(1, 0, 2)
        
        # # 4. Q-Former前向传播
        # for i in range(self.num_q_former_layers):
        #     meta_motion_queries = self.transformer_cross_attention_layers_1[i](meta_motion_queries, motion_feats, pos=pos_embed, query_pos=query_pos_embed)
        #     meta_motion_queries = self.transformer_self_attention_layers[i](meta_motion_queries, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos_embed)
        #     meta_motion_queries = self.transformer_cross_attention_layers_2[i](meta_motion_queries, appearance_feats, pos=pos_embed, query_pos=query_pos_embed)
        #     meta_motion_queries = self.transformer_ffn_layers[i](meta_motion_queries)
        
        # 最后阶段，使用对应的mask_head
        # 利用类似mask2former中的mask_head结构
        # 输入为meta_motion_prototype (num_meta_motion_query, b*n_way, c) 和 query_feats (b, T, c, h, w)
        # 输出为mask: (b, n_way, num_meta_motion_query, T, h, w)
        # 最终将按num_meta_motion_query相加变为 (b, n_way, T, h, w)
        B, T = query_feats.shape[:2]
        # query_feats = query_feats.view(B*T, *query_feats.shape[2:])
        # query_feats = self.query_proj(query_feats)
        # query_feats = query_feats.view(B, T, *query_feats.shape[1:])
        query_mask_feats = query_mask_feats.view(B, T, *query_mask_feats.shape[1:])
        mask = self.decoder(query_mask_feats, meta_motion_queries, query_pos_embed, query_feats_['ms_feats'])  # B, N_way, T, H, W
        
        return mask

class MaskFormerDecoder(nn.Module):
    """Mask2Former风格解码器"""
    def __init__(self, dim):
        super().__init__()
        self.hidden_dim = dim
        self.num_heads = 8
        self.num_layers = 6
        
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=256,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=256,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=256,
                    dim_feedforward=256*2,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
        
        self.pe_layer = PositionEmbeddingSine3D(self.hidden_dim // 2, normalize=True)
        self.decoder_norm = nn.LayerNorm(self.hidden_dim)
        
        # 预测mask的MLP头
        self.mask_embed = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )
        
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, self.hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if 256 != self.hidden_dim:
                self.input_proj.append(torch.nn.Conv2d(256, self.hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        
        self.mask_refiner1 = nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, kernel_size=3, padding=1)
        self.mask_refiner2 = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=3, padding=1)
        self.mask_predict = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        b = mask_embed.shape[0]
        q = mask_embed.shape[1]
        t = mask_features.shape[1]
        h, w = mask_features.shape[-2:]

        outputs_mask = mask_embed.unsqueeze(3).unsqueeze(4) * mask_features
        outputs_mask = outputs_mask.flatten(0, 1)
        # b, q, t, _, _ = outputs_mask.shape
        outputs_mask = self.mask_refiner1(outputs_mask)
        outputs_mask = self.mask_refiner2(outputs_mask)
        outputs_mask = self.mask_predict(outputs_mask)
        outputs_mask = outputs_mask.squeeze(1)
        outputs_mask = outputs_mask.view(b, t, h, w)
        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1).unsqueeze(1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_mask, attn_mask

    def forward(self, img_feats, query_feats, query_pos_embed, query_ms_feats):
        """
        Args:
            img_feats: (B, T, C, H, W)
            query_feats: (N, B*K, C) where N is num_meta_motion_query, K is n_way
        Returns:
            masks: (B, K, T, H, W)
        """
        B, T, C, H, W = img_feats.shape
        # query_feats = query_feats.transpose(0, 1)  # N, B*K, C
        N, BK, _ = query_feats.shape
        
        K = BK // B  # n_way
        
        # 重塑图像特征
        # img_feats = img_feats.view(B*T, C, H, W)  # B*
        
        # 将空间维度展平作为序列
        # img_feats = img_feats.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        # img_pe = self.pe_layer(img_feats)
        # img_pe = img_pe.unsqueeze(1).repeat(1, K, 1, 1, 1, 1)  # (B, N, T, C, H, W)
        # img_pe = einops.rearrange(img_pe, 'b k t c h w -> (t h w) (b k) c')  # (T*H*W, B*K, C)
        
        # img_feats = img_feats.unsqueeze(1).repeat(1, K, 1, 1, 1, 1)  # (B, N, T, C, H, W)
        # img_feats = einops.rearrange(img_feats, 'b k t c h w -> (t h w) (b k) c')  # (T*H*W, B*K, C)
        mask_features = img_feats
        
    
        # x is a list of multi-scale feature
        assert len(query_ms_feats) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(query_ms_feats[i].shape[-2:])
            pos.append(self.pe_layer(query_ms_feats[i].view(B, T, -1, size_list[-1][0], size_list[-1][1]), None).flatten(3))
            src.append(self.input_proj[i](query_ms_feats[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # NTxCxHW => NxTxCxHW => (TxHW)xNxC
            _, c, hw = src[-1].shape
            pos[-1] = pos[-1].view(B, T, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
            src[-1] = src[-1].view(B, T, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
        
        # 通过transformer layers
        output = query_feats  # (N, B*K, C)
        
        # for i in range(self.num_layers):
        #     tgt = self.transformer_cross_attention_layers[i](tgt, memory, pos=memory_pe, query_pos=query_pos_embed)  # (N, B*K, C)
        #     tgt = self.transformer_self_attention_layers[i](tgt, query_pos=query_pos_embed)
        #     tgt = self.transformer_ffn_layers[i](tgt)
        outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_pos_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_pos_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])

        # # 生成mask嵌入
        # tgt = self.decoder_norm(tgt)
        # mask_embed = self.mask_embed(tgt)  # (N, B*K, C)
        
        # # 重塑为最终输出格式
        # mask_embed = mask_embed.view(N, B, K, C)  # (N, B, K, C) 
        # mask_embed = mask_embed.permute(1, 2, 0, 3)  # (B, K, N, C)
        
        # # 重塑图像特征
        # img_feats = img_feats.permute(1, 2, 0)       # (B*K, C, T*H*W)
        # img_feats = img_feats.view(B, K, C, T, H*W)  # (B, K, C, T, H*W)
        # img_feats = img_feats.permute(0, 1, 3, 2, 4) # (B, K, T, C, H*W)
        
        # # 扩展mask_embed以匹配时序维度
        
        # # 计算相似度得到mask
        # masks = torch.einsum('bktcl,bknc->bkntl', img_feats, mask_embed)
        # masks = masks.view(B, K, N, T, H, W)
        # masks = masks[:, :, 0, :, :, :]
        
        # # 
        # masks = masks.sum(dim=2)  # (B, K, T, H, W)
        # masks = masks.clamp(0, 1)  # 将值截断到0和1之间
        masks = outputs_mask.view(B, K, T, H, W)
        
        return masks

class PixelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        input_shape = {
            'res2': ShapeSpec(channels=256, stride=4),
            'res3': ShapeSpec(channels=512, stride=8),
            'res4': ShapeSpec(channels=1024, stride=16),
            'res5': ShapeSpec(channels=2048, stride=32),
        }

        transformer_dropout = 0.0
        transformer_nheads = 8
        transformer_dim_feedforward = 1024
        transformer_enc_layers = 6
        conv_dim = 256
        mask_dim = 256
        norm = 'GN'
        transformer_in_features = ['res3', 'res4', 'res5']
        common_stride = 4
        
        self.msdeformattn_pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape=input_shape,
            transformer_dropout=transformer_dropout,
            transformer_nheads=transformer_nheads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_enc_layers=transformer_enc_layers,
            conv_dim=conv_dim,
            mask_dim=mask_dim,
            norm=norm,
            transformer_in_features=transformer_in_features,
            common_stride=common_stride,
        )
    
    def forward(self, features):
        features = self.msdeformattn_pixel_decoder.forward_features(features)
        return features
        
        

# 测试代码
if __name__ == "__main__":
    # # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_way = 1
    k_shot = 5
    num_support_frames = 5
    model = DMA(n_way=n_way, k_shot=k_shot, num_support_frames=num_support_frames, num_meta_motion_queries=1).to(device)
    
    # Create test data
    B, T = 2, 8  # batch size, num frames
    H, W = 241, 425  # height, width
    query_video = torch.randn(B, T, 3, H, W).to(device)
    support_video = torch.randn(B, n_way, k_shot, num_support_frames, 3, H, W).to(device)
    support_mask = torch.randn(B, n_way, k_shot, num_support_frames, H, W).to(device)
    
    # Forward pass
    masks = model(query_video, support_video, support_mask)
    print("Output masks shape:", masks.shape)
    
    # pixel_decoder = PixelDecoder()