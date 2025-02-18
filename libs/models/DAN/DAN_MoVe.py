#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import libs.models.DAN.resnet as models
from libs.models.DAN.decoder import Decoder

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)

        self.layer1 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024 --> 1/8

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, in_f):
        f = in_f
        x = self.layer0(f)
        l1 = self.layer1(x)  # 1/4, 256
        l2 = self.layer2(l1)  # 1/8, 512
        l3 = self.layer3(l2)  # 1/8, 1024

        return l3, l3, l2, l1

class QueryKeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(QueryKeyValue, self).__init__()
        self.query = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.query(x), self.Key(x), self.Value(x)

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super(TemporalAttention, self).__init__()
        self.temporal_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.temporal_attn = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim//4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        temporal_feat = self.temporal_embed(x)
        attn = self.temporal_attn(temporal_feat)
        x = x * attn
        return x.view(B, T, C, H, W)

class DAN_MoVe(nn.Module):
    def __init__(self, support_frames=15, query_frames=1):
        super(DAN_MoVe, self).__init__()
        self.encoder = Encoder()  # output 1024
        encoder_dim = 1024
        h_encdim = int(encoder_dim/2)
        self.support_qkv = QueryKeyValue(encoder_dim, keydim=128, valdim=h_encdim)
        self.query_qkv = QueryKeyValue(encoder_dim, keydim=128, valdim=h_encdim)
        
        self.conv_q = nn.Conv2d(encoder_dim, h_encdim, kernel_size=1, stride=1, padding=0)
        self.Decoder = Decoder(encoder_dim, 256)
        
        self.support_frames = support_frames
        self.query_frames = query_frames

    def transformer(self, Q, K, V):
        # Q : B CQ WQ
        # K : B WK CQ
        # V : B CV WK
        B, CQ, WQ = Q.shape
        _, CV, WK = V.shape

        P = torch.bmm(K, Q)  # B WK WQ
        P = P / math.sqrt(CQ)
        P = torch.softmax(P, dim=1)

        M = torch.bmm(V, P)  # B CV WQ

        return M, P

    def forward(self, img, support_image, support_mask, time=None):
        batch, frame, in_channels, height, width = img.shape
        batch_shot, sframe, mask_channels, Sheight, Swidth = support_mask.shape
        assert height == Sheight and width == Swidth
        
        # Reshape inputs for encoder
        batch_frame = batch * frame
        img = img.view(-1, in_channels, height, width)
        support_image = support_image.view(-1, in_channels, height, width)
        support_mask = support_mask.view(-1, mask_channels, height, width)
        
        # Encode all frames
        in_f = torch.cat((img, support_image), dim=0)
        encoder_f, encoder_f_l3, encoder_f_l2, encoder_f_l1 = self.encoder(in_f)
        
        # Split features
        query_feat = encoder_f[:batch_frame]
        support_feat = encoder_f[batch_frame:]
        
        # Get query features at different levels
        query_feat_l1 = encoder_f_l1[:batch_frame]
        query_feat_l2 = encoder_f_l2[:batch_frame]
        
        # Upsample support mask
        support_mask = F.interpolate(support_mask, encoder_f.size()[2:], mode='bilinear', align_corners=True)
        
        # Extract foreground and background features
        support_fg_feat = support_feat * support_mask
        support_bg_feat = support_feat * (1 - support_mask)
        
        # Generate QKV
        _, support_k, support_v = self.support_qkv(support_fg_feat)
        query_q, query_k, query_v = self.query_qkv(query_feat)
        
        # Get dimensions
        _, c, h, w = support_k.shape
        _, vc, _, _ = support_v.shape
        
        # Reshape support features to (batch, num_shots, frames_per_shot, channels, height, width)
        num_shots = batch_shot // batch
        support_k = support_k.view(batch, num_shots, sframe, c, h, w)
        support_v = support_v.view(batch, num_shots, sframe, vc, h, w)
        
        # Merge shots and frames dimensions for transformer
        # B, (S*F), C, H, W -> B, C, (S*F*H*W)
        support_k = support_k.view(batch, num_shots * sframe, c, h, w)
        support_k = support_k.permute(0, 2, 1, 3, 4).contiguous()
        support_k = support_k.view(batch, c, -1).permute(0, 2, 1).contiguous()  # B, WK, CK
        
        support_v = support_v.view(batch, num_shots * sframe, vc, h, w)
        support_v = support_v.permute(0, 2, 1, 3, 4).contiguous()
        support_v = support_v.view(batch, vc, -1)  # B, CV, WK
        
        # Process middle frame
        middle_frame_index = int(frame/2)
        query_q = query_q.view(batch, frame, c, h, w)
        query_k = query_k.view(batch, frame, c, h, w)
        middle_q = query_q[:, middle_frame_index]
        middle_q = middle_q.view(batch, c, -1)
        
        # Cross-attention between query and support
        new_V, sim_refer = self.transformer(middle_q, support_k, support_v)
        
        # Self-attention within query frames
        middle_K = query_k[:, middle_frame_index]
        middle_K = middle_K.view(batch, c, -1).permute(0, 2, 1).contiguous()
        
        query_q = query_q.permute(0, 2, 1, 3, 4).contiguous().view(batch, c, -1)
        Out, sim_middle = self.transformer(query_q, middle_K, new_V)
        
        # Reshape and merge features
        after_transform = Out.view(batch, vc, frame, h, w)
        after_transform = after_transform.permute(0, 2, 1, 3, 4).contiguous()
        after_transform = after_transform.view(-1, vc, h, w)
        
        # Process query features
        query_feat = self.conv_q(query_feat)
        after_transform = torch.cat((after_transform, query_feat), dim=1)
        
        # Decode
        x = self.Decoder(after_transform, query_feat_l2, query_feat_l1, img)
        # pred_map = torch.sigmoid(x)
        pred_map = x
        
        # Reshape output
        pred_map = pred_map.view(batch, frame, 1, height, width)
        return pred_map

if __name__ == '__main__':
    # Test the model
    model = DAN_MoVe(support_frames=15, query_frames=3)
    img = torch.FloatTensor(2, 3, 3, 224, 224)
    support_mask = torch.FloatTensor(2, 15, 1, 224, 224)
    support_img = torch.FloatTensor(2, 15, 3, 224, 224)
    pred_map = model(img, support_img, support_mask)
    print(pred_map.shape) 