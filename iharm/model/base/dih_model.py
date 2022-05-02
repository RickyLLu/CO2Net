import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from iharm.model.modeling.conv_autoencoder import ConvEncoder, DeconvDecoder

class MyAttnByMask(nn.Module):
    def __init__(self, in_dim):
        super(MyAttnByMask, self).__init__()
        self.query_proj = nn.Conv2d(in_dim, in_dim, kernel_size = 1)
        self.key_proj = nn.Conv2d(in_dim, in_dim, kernel_size = 1)
        self.value_proj = nn.Conv2d(in_dim, in_dim, kernel_size = 1)
        self.out_proj = nn.Conv2d(in_dim, in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, mask, pre_feat = None):
        ##input and pre_feat is b, c, w, h
        #query = self.query_proj(input)
        b, c, w, h = input.shape
        mb, mc, mw, mh = mask.shape
        if h!=mh:
            tmp_mask = torch.round(F.avg_pool2d(mask, 2, stride=mh//h))
        else:
            tmp_mask = copy.copy(mask)
        reverse_mask = 1.0 - tmp_mask
        curr_feat_f = input * tmp_mask
        curr_feat_b = input * reverse_mask
        query = self.query_proj(curr_feat_f)
        if pre_feat:

            key_1 = self.value_proj(curr_feat_b)
            key_2 = self.value_proj(pre_feat)
            val_1 = self.key_proj(curr_feat_b)
            val_2 = self.key_proj(pre_feat)
            key = torch.cat((key_1, key_2), axis = 2)
            val = torch.cat((val_1, val_2), axis = 2)
        else:
            key = self.key_proj(curr_feat_b)
            val = self.value_proj(curr_feat_b)
        query = query.reshape((b ,c, -1)).permute(0, 2, 1)
        key = key.reshape((b, c, -1))
        val = val.reshape((b, c, -1)).permute(0, 2, 1)
        scores = torch.bmm(query, key)
        attention = self.softmax(scores)
        out = torch.bmm(attention, val).permute(0, 2, 1)
        out = self.out_proj(out.view(b, c, w, h))
        return out + self.gamma*input



class DeepImageHarmonization(nn.Module):
    def __init__(
        self,
        depth,
        norm_layer=nn.BatchNorm2d, batchnorm_from=0,
        attend_from=-1,
        image_fusion=False,
        ch=64, max_channels=512,
        backbone_from=-1, backbone_channels=None, backbone_mode=''
    ):
        super(DeepImageHarmonization, self).__init__()
        self.depth = depth
        self.encoder = ConvEncoder(
            depth, ch,
            norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        self.attn = MyAttnByMask(max_channels)
        self.decoder = DeconvDecoder(depth, self.encoder.blocks_channels, norm_layer, attend_from, image_fusion)

    def forward(self, image, mask, backbone_features=None, previous_feat = None):
        x = torch.cat((image, mask), dim=1)
        intermediates = self.encoder(x, backbone_features)
        #previous_feat = torch.randn((2, 512 ,1 ,1)).cuda()
        for hidden_feat in intermediates:
            print(hidden_feat.shape)


        intermediates[0] = self.attn(intermediates[0], mask, previous_feat)

        output = self.decoder(intermediates, image, mask)
        exit()
        return {'images': output, 'cur_feat':intermediates[0]}
