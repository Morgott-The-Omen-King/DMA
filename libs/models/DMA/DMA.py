import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import math
import einops


class DMA(nn.Module):
    """Decoupled Motion-Appearance Network"""
    def __init__(self, 
                 n_way=2, 
                 k_shot=1, 
                 num_support_frames=5, 
                 backbone='resnet50',
                 hidden_dim=256):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.num_support_frames = num_support_frames
        
        # q-former
        self.num_meta_motion_queries = 20
    
        self.build_backbone()
            
        # 特征维度
        self.hidden_dim = hidden_dim
        
        # Motion-Appearance解耦模块
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(self.feat_dim, self.hidden_dim, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.appear_encoder = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
            
        # Q-Former风格的动作分解模块
        # Meta-motion queries
        self.meta_motion_queries = nn.Parameter(torch.zeros(1, self.num_meta_motion_queries, self.hidden_dim))
        self.query_pos_embed = nn.Parameter(torch.zeros(1, self.num_meta_motion_queries, self.hidden_dim))
        
        # 定义多层Transformer结构
        self.motion_transformer = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(self.hidden_dim),  # Cross-attention 1
                nn.LayerNorm(self.hidden_dim),
                MultiHeadAttention(self.hidden_dim),  # Cross-attention 2
                nn.LayerNorm(self.hidden_dim),
                MultiHeadAttention(self.hidden_dim),  # Self-attention
                nn.LayerNorm(self.hidden_dim),
                nn.Sequential(  # FFN
                    nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim)
                ),
                nn.LayerNorm(self.hidden_dim)
            )
            for _ in range(6)  # 6层
        ])
        
        # mask_feats head
        # 1x1 convolutions for adjusting channel dimensions
        self.layer2_conv = nn.Conv2d(self.feat_dims['layer2'], self.hidden_dim, 1)
        self.layer3_conv = nn.Conv2d(self.feat_dims['layer3'], self.hidden_dim, 1)
        
        # Final 1x1 conv for fusing multi-scale features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 特征映射层
        self.motion_proj = nn.Linear(self.feat_dim, self.hidden_dim)
        self.appear_proj = nn.Linear(self.feat_dim, self.hidden_dim)
        
        self.query_proj = nn.Conv2d(self.feat_dim, self.hidden_dim, kernel_size=1)
        
        # 分割头
        self.decoder = MaskFormerDecoder(self.hidden_dim)
        
        self.mask_head = nn.Conv2d(self.hidden_dim, self.n_way, kernel_size=1)

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
        nn.init.normal_(self.meta_motion_queries, std=0.02)
        nn.init.normal_(self.query_pos_embed, std=0.02)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def build_backbone(self):
        """构建骨干网络"""
        # 加载预训练的ResNet50
        backbone = resnet50(pretrained=True)
        
        # 获取不同层的特征
        self.layer1 = nn.Sequential(*list(backbone.children())[:5])  # 1/4
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
        
    def extract_features(self, rgb_input, mask_feats=False):
        """使用骨干网络提取多尺度特征"""
        # 提取多尺度特征
        feat_1_4 = self.layer1(rgb_input)
        feat_1_8 = self.layer2(feat_1_4)
        feat_1_16 = self.layer3(feat_1_8)
        feat_1_32 = self.layer4(feat_1_16)
        
        # 返回多尺度特征字典
        features = {
            'layer1': feat_1_4,
            'layer2': feat_1_8,
            'layer3': feat_1_16,
            'layer4': feat_1_32
        }
        
        if mask_feats:
            # 将layer2和layer3上采样到layer1尺度
            feat_1_8_up = F.interpolate(
                self.layer2_conv(feat_1_8),  # 1x1 conv to adjust channels
                size=feat_1_4.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            feat_1_16_up = F.interpolate(
                self.layer3_conv(feat_1_16),  # 1x1 conv to adjust channels 
                size=feat_1_4.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # 将多尺度特征相加并通过1x1卷积融合
            mask_feats = self.fusion_conv(feat_1_4 + feat_1_8_up + feat_1_16_up)
            features['mask_feats'] = mask_feats
            
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
        query_feats = self.extract_features(query_video.reshape(B*T, C, H, W), mask_feats=True)  # todo:这个地方可能需要提取多尺度的特征
        
        query_mask_feats, query_feats = query_feats['mask_feats'], query_feats['mask_feats']
        
        feat_h, feat_w = query_mask_feats.shape[-2:]
        # # Convert query mask features to n-way classification mask
        # query_mask_feats = query_mask_feats.view(B*T, self.hidden_dim, feat_h, feat_w)
        # query_mask_logits = self.mask_head(query_mask_feats)  # Convert to n-way logits
        # query_mask_logits = query_mask_logits.view(B, T, self.n_way, feat_h, feat_w)
        
        # # Upsample to original resolution
        # query_mask_logits = F.interpolate(
        #     query_mask_logits.view(B*T, self.n_way, feat_h, feat_w),
        #     size=(H, W),
        #     mode='bilinear',
        #     align_corners=False
        # )
        # query_mask_logits = query_mask_logits.view(B, T, self.n_way, H, W).sigmoid()
        # query_mask_logits = query_mask_logits.permute(0,2,1,3,4)
        # return query_mask_logits
        
        
        feat_h, feat_w = query_feats.shape[-2:]  # 获取特征图的实际高宽
        query_feats = query_feats.view(B, T, self.feat_dim, feat_h, feat_w)
        
        # Support视频特征
        support_feats = self.extract_features(support_video.view(-1, C, H, W), mask_feats=True)
        support_mask_feats, support_feats = support_feats['mask_feats'], support_feats['mask_feats']
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
        masked_feats = support_feats_flat * support_mask_resized
        
        # 使用全局平均池化将特征压缩为向量
        masked_feats = F.adaptive_avg_pool2d(masked_feats, 1).squeeze(-1).squeeze(-1)
        appearance_feats = masked_feats.view(B, self.n_way, self.k_shot, self.num_support_frames, self.feat_dim)
        
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
        motion_feats = motion_feats.view(B * self.n_way, self.k_shot * self.num_support_frames, self.hidden_dim).permute(1, 0, 2)
        appearance_feats = appearance_feats.view(B * self.n_way, self.k_shot * self.num_support_frames, self.hidden_dim).permute(1, 0, 2)
        
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
        
        # 3. 初始化meta-motion queries
        meta_motion_queries = self.meta_motion_queries.expand(B * self.n_way, -1, -1)
        query_pos_embed = self.query_pos_embed.expand(B * self.n_way, -1, -1)
        
        # 4. Q-Former前向传播
        for layer in self.motion_transformer:  # 遍历6层Transformer
            # 获取当前层的各个模块
            cross_attn1, norm1, cross_attn2, norm2, self_attn, norm3, ffn, norm4 = layer
            
            # 第一层Cross-attention: 与motion特征交互
            motion_attn = cross_attn1(
                meta_motion_queries + query_pos_embed,
                motion_feats + pos_embed,
                motion_feats + pos_embed
            )
            motion_attn = norm1(motion_attn)
            
            # 第二层Cross-attention: 与appearance特征交互
            appear_attn = cross_attn2(
                motion_attn + query_pos_embed,
                appearance_feats + pos_embed,
                appearance_feats + pos_embed
            )
            appear_attn = norm2(appear_attn)
            
            # Self-attention: meta-motion queries自身交互
            meta_motion_queries = self_attn(
                appear_attn + query_pos_embed,
                appear_attn + query_pos_embed,
                appear_attn + query_pos_embed
            )
            meta_motion_queries = norm3(meta_motion_queries)
            
            # FFN
            meta_motion_queries = ffn(meta_motion_queries)
            meta_motion_queries = norm4(meta_motion_queries)
        
        # 5. 获取meta-motion prototype
        meta_motion_prototype = meta_motion_queries.view(B, self.n_way, self.num_meta_motion_queries, self.hidden_dim)
        
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
        mask = self.decoder(query_mask_feats, meta_motion_queries)  # B, N_way, T, H, W
        
        resize_mask = F.interpolate(mask.flatten(0,1), size=(H, W), mode='bilinear', align_corners=False).view(B, self.n_way, T, H, W).sigmoid()  # B n t h w
        
        return resize_mask


class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, q, k, v):
        B, N, C = q.shape
        
        qkv = self.qkv(q).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MaskFormerDecoder(nn.Module):
    """Mask2Former风格解码器"""
    def __init__(self, dim):
        super().__init__()
        self.hidden_dim = dim
        self.num_heads = 8
        self.num_layers = 6
        
        # 多层transformer decoder
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads
            ) for _ in range(self.num_layers)
        ])
        
        # 预测mask的MLP头
        self.mask_embed = MLP(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            num_layers=2
        )
        
    def forward(self, img_feats, query_feats):
        """
        Args:
            img_feats: (B, T, C, H, W)
            query_feats: (N, B*K, C) where N is num_meta_motion_query, K is n_way
        Returns:
            masks: (B, K, T, H, W)
        """
        B, T, C, H, W = img_feats.shape
        query_feats = query_feats.transpose(0, 1)  # N, B*K, C
        N, BK, _ = query_feats.shape
        
        K = BK // B  # n_way
        
        # 重塑图像特征
        # img_feats = img_feats.view(B*T, C, H, W)  # B*
        
        # 将空间维度展平作为序列
        # img_feats = img_feats.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        img_feats = img_feats.unsqueeze(1).repeat(1, K, 1, 1, 1, 1)  # (B, N, T, C, H, W)
        img_feats = einops.rearrange(img_feats, 'b k t c h w -> (t h w) (b k) c')  # (T*H*W, B*K, C)
    
        
        # 通过transformer layers
        tgt = query_feats  # (N, B*K, C)
        memory = img_feats  # (T*H*W, B*K, C)
        
        for layer in self.transformer_layers:
            tgt = layer(tgt, memory)  # (N, B*K, C)
        
        # 生成mask嵌入
        mask_embed = self.mask_embed(tgt)  # (N, B*K, C)
        
        # 重塑为最终输出格式
        mask_embed = mask_embed.view(N, B, K, C)  # (N, B, K, C) 
        mask_embed = mask_embed.permute(1, 2, 0, 3)  # (B, K, N, C)
        
        # 重塑图像特征
        img_feats = img_feats.permute(1, 2, 0)  # (B*K, C, T*H*W)
        img_feats = img_feats.view(B, K, C, T, H*W)  # (B, K, C, T, H*W)
        img_feats = img_feats.permute(0, 1, 3, 2, 4)  # (B, T, C, H*W)
        
        # 扩展mask_embed以匹配时序维度
        
        # 计算相似度得到mask
        masks = torch.einsum('bktcl,bknc->bkntl', img_feats, mask_embed)
        masks = masks.view(B, K, N, T, H, W) 
        
        # 
        masks = masks.sum(dim=2)  # (B, K, T, H, W)
        
        return masks

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 多头交叉注意力
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
    def forward(self, tgt, memory):
        # 自注意力
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 测试代码
if __name__ == "__main__":
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMA(n_way=2, k_shot=1, num_support_frames=10).to(device)
    
    # 创建测试数据
    B, T = 2, 16
    H, W = 241, 425
    query_video = torch.randn(B, T, 3, H, W).to(device)
    support_video = torch.randn(B, 2, 1, 10, 3, H, W).to(device)
    support_mask = torch.randn(B, 2, 1, 10, H, W).to(device)
    
    # 前向传播
    masks = model(query_video, support_video, support_mask)
    print("Output masks shape:", masks.shape)
