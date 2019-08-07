import torch
import torch.nn as nn
import utils as utils
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("instantiating " + self.__class__.__name__)

    def forward(self, x, b, t):
        return [x, None, None]


class COSAM(nn.Module):
    def __init__(self, in_channels, t, h_w, *args):
        super().__init__()
        print(
            "instantiating "
            + self.__class__.__name__
            + " with in-channels: "
            + str(in_channels)
            + ", t = "
            + str(t)
            + ", hw = "
            + str(h_w)
        )

        self.eps = 1e-4
        self.h_w = h_w
        self.t = t
        self.mid_channels = 256
        if in_channels <= 256:
            self.mid_channels = 64

        self.dim_reduction = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
        )

        self.spatial_mask_summary = nn.Sequential(
            nn.Conv2d((t - 1) * h_w[0] * h_w[1], 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.channelwise_attention = nn.Sequential(
            nn.Linear(in_channels, self.mid_channels),
            nn.Tanh(),
            nn.Linear(self.mid_channels, in_channels),
            nn.Sigmoid(),
        )

    def get_selectable_indices(self, t):
        init_list = list(range(t))
        index_list = []
        for i in range(t):
            list_instance = list(init_list)
            list_instance.remove(i)
            index_list.append(list_instance)

        return index_list

    def get_channelwise_attention(self, feat_maps, b, t):
        num_imgs, num_channels, h, w = feat_maps.shape

        # perform global average pooling
        channel_avg_pool = F.adaptive_avg_pool2d(feat_maps, output_size=1)
        # pass the global average pooled features through the fully connected network with sigmoid activation
        channelwise_attention = self.channelwise_attention(channel_avg_pool.view(num_imgs, -1))

        # perform group attention
        # groupify the attentions
        idwise_channelattention = channelwise_attention.view(b, t, -1)
        
        # take the mean of attention to attend common channels between frames
        group_attention = torch.mean(idwise_channelattention, dim=1, keepdim=True).expand_as(idwise_channelattention)
        channelwise_attention = group_attention.contiguous().view(num_imgs, num_channels, 1, 1)

        return channelwise_attention

    def get_spatial_attention(self, feat_maps, b, t):
        total, c, h, w = feat_maps.shape
        dim_reduced_featuremaps = self.dim_reduction(feat_maps)  # #frames x C x H x W

        # resize the feature maps for temporal processing
        identitywise_maps = dim_reduced_featuremaps.view(b, t, dim_reduced_featuremaps.shape[1], h, w)
        
        # get the combination of frame indices 
        index_list = self.get_selectable_indices(t)

        # select the other images within same id
        other_selected_imgs = identitywise_maps[:, index_list]
        other_selected_imgs = other_selected_imgs.view(-1, t - 1, dim_reduced_featuremaps.shape[1], h, w)  # #frames x t-1 x C x H x W

        # permutate the other dimensions except descriptor dim to last
        other_selected_imgs = other_selected_imgs.permute((0, 2, 1, 3, 4))  # #frames x C x t-1 x H x W

        # prepare two matrices for multiplication
        dim_reduced_featuremaps = dim_reduced_featuremaps.view(total, self.mid_channels, -1)  # #frames x C x (H * W)
        dim_reduced_featuremaps = dim_reduced_featuremaps.permute((0, 2, 1))  # frames x (H * W) x C
        other_selected_imgs = other_selected_imgs.contiguous().view(total, self.mid_channels, -1)  # #frames x C x (t-1 * H * W)

        # mean subtract and divide by std
        dim_reduced_featuremaps = dim_reduced_featuremaps - torch.mean(dim_reduced_featuremaps, dim=2, keepdim=True)
        dim_reduced_featuremaps = dim_reduced_featuremaps / (torch.std(dim_reduced_featuremaps, dim=2, keepdim=True) + self.eps)
        other_selected_imgs = other_selected_imgs - torch.mean(other_selected_imgs, dim=1, keepdim=True)
        other_selected_imgs = other_selected_imgs / (torch.std(other_selected_imgs, dim=1, keepdim=True) + self.eps)
        
        mutual_correlation = (torch.bmm(dim_reduced_featuremaps, other_selected_imgs)
                                        / other_selected_imgs.shape[1])  # #frames x (HW) x (t-1 * H * W)

        mutual_correlation = mutual_correlation.permute(0, 2, 1)  # #frames x (t-1 * H * W) x (HW)
        mutual_correlation = mutual_correlation.view(total, -1, h, w)  # #frames x (t-1 * H * W) x H x W
        mutual_correlation_mask = self.spatial_mask_summary(mutual_correlation).sigmoid() # #frames x 1 x H x W
        
        return mutual_correlation_mask

    def forward(self, feat_maps, b, t):
        # get the spatial attention mask
        mutualcorr_spatial_mask = self.get_spatial_attention(
            feat_maps=feat_maps, b=b, t=t
        )
        attended_out = torch.mul(feat_maps, mutualcorr_spatial_mask)

        # channel-wise attention
        channelwise_mask = self.get_channelwise_attention(
            feat_maps=attended_out, b=b, t=t
        )
        attended_out = attended_out + torch.mul(attended_out, channelwise_mask)

        return attended_out, channelwise_mask, mutualcorr_spatial_mask


COSEG_ATTENTION = {
    "NONE": Identity,
    "COSAM": COSAM,
}
