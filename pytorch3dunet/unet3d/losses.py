import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np
from pytorch3dunet.unet3d.utils import expand_as_one_hot
from skimage.segmentation import boundaries

def compute_per_channel_dice_old(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):

    # input and target shapes must match
    assert weight is None
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # input = flatten(input)
    # target = flatten(target)
    input = input.float()
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum()
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = input.sum() + target.sum()
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())





class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()





def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config, key_name="loss", weight=None, pos_weight=None):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config[key_name]
    name = loss_config.get('name')

    ignore_index = loss_config.get('ignore_index', None)
    assert ignore_index is None
    skip_last_target = loss_config.get('skip_last_target', False)
    assert not skip_last_target

    if weight is not None:
        if isinstance(weight, np.ndarray):
            weight = torch.tensor(weight)
        # convert to cuda tensor if necessary
        weight = weight.to(config['device'])

    if pos_weight is not None:
        if not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight)
        # convert to cuda tensor if necessary
        pos_weight = pos_weight.to(config['device'])

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    # if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
    #     # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
    #     loss = _MaskingLossWrapper(loss, ignore_index)

    # if skip_last_target:
    #     loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    return loss


#######################################################################################################################


class WeightedDiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight, pos_weight=None):
        return compute_per_channel_dice(input, target, weight=weight)

class WeightedBCEWithLogitsLoss(nn.Module):
    # this loss also serves as the distillation loss in segmentation case
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self):
        super(WeightedBCEWithLogitsLoss, self).__init__()

    def forward(self, input, target, weight, pos_weight, is_keep_batch_dim=False):
        """is_keep_batch_dim: if True, then we keep the batch dimension and take mean over image dimensions
        we assume the volume should be (batch, h, w, d)
        """
        if is_keep_batch_dim:
            # raise ValueError(f"Check the actual size of input! {input.size()}")
            return F.binary_cross_entropy_with_logits(input, target, weight=weight, pos_weight=pos_weight, reduction="none").mean(dim=(1,2,3,4))
        return F.binary_cross_entropy_with_logits(
            input, target, weight=weight, pos_weight=pos_weight
        )

class DistillationLoss(nn.Module):
    """though WeightedBCEWithLogitsLoss serve as distillation loss, we add temporature in this loss"""
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, input, target, weight, pos_weight, is_keep_batch_dim=False):
        
        target = target.detach()
        # apply temporature
        # logits divided by temporature
        # probabilities take the power of 1 / temporature
        if self.temperature is not None:
            # input is logits
            input = torch.div(input, self.temperature)
            # target is probability/1 or 0
            target = torch.pow(target, 1.0 / self.temperature)
        if is_keep_batch_dim:
            # raise ValueError(f"Check the actual size of input! {input.size()}")
            return F.binary_cross_entropy_with_logits(input, target, weight=weight, pos_weight=pos_weight, reduction="none").mean(dim=(1,2,3,4))
        return F.binary_cross_entropy_with_logits(
            input, target, weight=weight, pos_weight=pos_weight
        )

# implement contrastive distillation loss
class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning for segmentation
    the reward for contrastive loss for medical image segmentation (different from semantic segmentation)
    those class-incremental setting does not take care of overlap class (same class, but the new groud truth and old model's prediction could disagree)
    
    Here are different cases depending on locations of the anchor and the pixel: note that anchor is never selected from the background (due to diversity of background, we cannot force similarity)
    if gt[ac] == 1 and gt[px] == 1: then +(new[ac], new[px])
    if gt[ac] == 1 and gt[px] == 0: then -(new[ac], new[px])
    if gt[ac] == 1 and gt[px] == 1 and old[px] == 1: then + (new[ac], old[px])
    if gt[ac] == 1 and gt[px] == 1 and old[px] == 0: then - (new[ac], old[px])
    if gt[ac] == 1 and gt[px] == 0 and old[px] == 0: then - (new[ac], old[px])
    if gt[ac] == 1 and gt[px] == 0 and old[px] == 1: then - (new[ac], old[px])
    """

    def __init__(self, device, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.device = device
        self.temperature = temperature

    def forward(self, anchor_features, contrast_feature, contrib_mask_pos, contrib_mask_neg, conf_mask=None):
        """
        Args:
            achor_features: hidden vector of shape [bsz, 256].
            contrast_features: hidden vector of shape [bsz_prime, 256].
            contrib_mask: [bsz, bsz + bsz_prime], positive contrastive contribution 1, negative 0
            conf_mask: shape [bsz, bsz + bsz_prime], the uncertainty level, 1.0 means most certain, 0.0 most uncertain

            bsz is anchor size (total number of anchor pixels in the batch)
            bsz and bsz_prime involves batch, h, w, d (batch and spatial dimensions)

        Returns:
            A loss scalar.
        """
        if anchor_features.shape[0] == 0 or contrast_feature.shape[0] == 0:
            return None
        if contrib_mask_pos.sum().item() == 0:
            return None
        num = contrib_mask_pos.sum(dim=1)
        if (num != 0).sum().item() == 0:
            # no loss
            return None

        # bsz by bsz_prime
        anchor_dot_contrast = torch.div(
            torch.mm(anchor_features, contrast_feature.T),
            self.temperature)

        ### compute negtive : anchor*R-
        # neg_contrast is of size [bsz, 1], we are suming over all negative locations on bsz + bsz_prime!
        # 
        
        # for numerical stability, globally subtract out same value does not matter
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - logits_max.detach()
        # we move this down here to keep the consistence
        neg_contrast = (torch.exp(anchor_dot_contrast) * contrib_mask_neg).sum(dim=1,keepdim=True)
        if conf_mask is None:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * contrib_mask_pos - torch.log(
                torch.exp(anchor_dot_contrast) + neg_contrast) * contrib_mask_pos
        else:
            pos_contrast = torch.log(torch.exp(anchor_dot_contrast)) * contrib_mask_pos * conf_mask - torch.log(
                torch.exp(anchor_dot_contrast) + neg_contrast) * contrib_mask_pos * conf_mask
        
        loss = -torch.div(pos_contrast.sum(dim=1)[num != 0], num[num != 0])
        return loss.mean()

# contrastive loss helper function - generate the input for contrastive loss
def get_binary_conf_(p):
    """compute binary confidence (the un-uncertainty)
    p is torch.tensor
    """
    if p.numel() == 0:
        return torch.tensor([]).reshape(p.shape).to(p.dtype).to(p.device)
    assert p.min() >= 0.0 and p.max() <= 1.0
    return p * p + (1.0 - p) * (1.0 - p)

def pre_contractive_pixel(f_n, l_n, f_o, l_po, is_edge_only=False):
    """
        f_n: new model's features on current image, features['pre_logits'], size batch, channel, h, w, d, in our chase
        l_n: new gt labels on current image, labels_, should be the true labels of the original resolution, size batch, 1, h_orig, w_orig, d_orig in our case, label value is either 0.0 or 1.0
        f_o: old model's features on current image, features_old['pre_logits'],batch, channel, h, w, d
        l_po: old model's segmentation prediction PROBABILITY (nn.Sigmoid(logits)) on current iimage, features_old['sem'], batch, 1, h_orig, w_orig, d_orig
        is_edge_only: the returned pos and neg mask only covers anchors and pixels around target boundaries

        we should detach from any old model's head, but keep grad in new models!
    """
    
    B, N, h, w, d = f_n.size()
    out_size = (h, w, d)
    ### for vis    
    mask_n_boundary = None
    if is_edge_only:
        mask_n_boundary = []
        # extract boundary pixels
        l_n_temp_ = l_n.detach().to("cpu").numpy().astype(np.int32)
        for ib in range(B):        
            m_bd = boundaries.find_boundaries(l_n_temp_[ib, 0, :, :, :], connectivity=3)
            m_bd = torch.from_numpy(m_bd).to(torch.float32).to(f_n.device).unsqueeze(dim=0).unsqueeze(dim=0)
            mask_n_boundary.append(m_bd)
        mask_n_boundary = torch.cat(mask_n_boundary, dim=0)
        mask_n_boundary = F.interpolate(mask_n_boundary, size=out_size, mode="trilinear", align_corners=False).type(torch.int8)
        mask_n_boundary = mask_n_boundary.squeeze(dim=1).reshape(B * h * w * d) # B, h, w, d
            
    label_n = F.interpolate(l_n, size=out_size, mode="trilinear", align_corners=False).type(torch.int8)
    label_n = label_n.squeeze(dim=1).reshape(B * h * w * d) # B, h, w, d
    l_po_ = F.interpolate(l_po.detach(), size=out_size, mode="trilinear", align_corners=False).type(torch.float32)    
    l_po_ = l_po_.squeeze(dim=1).reshape(B * h * w * d)
    label_po = (l_po_ > 0.5).to(torch.float32)
    ### without background
    # ALL non-background points in the reduced resolution (h by w) is treated as anchor "pixel"!   
    if is_edge_only:
        is_filtered = mask_n_boundary > 0
    else:
        is_filtered = True

    is_gt_p = (label_n > 0) & is_filtered
    is_gt_n = (label_n == 0) & is_filtered
    is_gt_p_op = (label_n > 0) & (label_po > 0) & is_filtered
    is_gt_p_on = (label_n > 0) & (label_po == 0) & is_filtered
    is_gt_n_op = (label_n == 0) & (label_po > 0) & is_filtered
    is_gt_n_on = (label_n == 0) & (label_po == 0) & is_filtered

    # print(f_n.size())
    f_n = f_n.permute(0, 2, 3, 4, 1)
    f_n = f_n.reshape(B, h * w * d, N)
    f_n = f_n.reshape(B * h * w * d, N)

    f_o = f_o.detach().permute(0, 2, 3, 4, 1) # B, h, w, d, channel
    f_o = f_o.reshape(B, h * w * d, N)
    f_o = f_o.reshape(B * h * w * d, N)

    # ALL non-background points in the reduced resolution (h by w) is sampled as anchor points, and thus anchor labels (N elements per pixel)
    Output_anchor = F.normalize(f_n[is_gt_p, :], dim=1)
    # corresponding to the organization of Lable_contrast
    Output_contrast = torch.cat(
        (
            F.normalize(f_n[is_gt_p, :], dim=1),
            F.normalize(f_n[is_gt_n, :], dim=1),
            F.normalize(f_o[is_gt_p_op, :], dim=1),
            F.normalize(f_o[is_gt_p_on, :], dim=1),
            F.normalize(f_o[is_gt_n_op, :], dim=1),
            F.normalize(f_o[is_gt_n_on, :], dim=1),
        ),
        dim=0,
    ).detach()

    # comfidence (un-uncertainty) matrix
    Conf_anchor = torch.ones(is_gt_p.sum(), 1).to(torch.float32).to(f_n.device) # bsz by 1
    Conf_contrast = torch.cat(
        (
            torch.ones(is_gt_p.sum(), 1).to(torch.float32).to(f_n.device),
            torch.ones(is_gt_n.sum(), 1).to(torch.float32).to(f_n.device),
            get_binary_conf_(l_po_[is_gt_p_op]).unsqueeze(dim=1),
            get_binary_conf_(l_po_[is_gt_p_on]).unsqueeze(dim=1),
            get_binary_conf_(l_po_[is_gt_n_op]).unsqueeze(dim=1),
            get_binary_conf_(l_po_[is_gt_n_on]).unsqueeze(dim=1),
        ),
        dim=0,
    ) # bsz_prime by 1
    Conf_matrix = torch.mm(Conf_anchor, Conf_contrast.T)
    
    # Contrib_matrix
    Contrib_anchor = torch.ones(is_gt_p.sum(), 1).to(torch.float32).to(f_n.device) # bsz by 1
    Contrib_contrast = torch.cat(
        (
            torch.ones(is_gt_p.sum(), 1).to(torch.float32).to(f_n.device),
            torch.zeros(is_gt_n.sum(), 1).to(torch.float32).to(f_n.device),
            torch.ones(is_gt_p_op.sum(), 1).to(torch.float32).to(f_n.device),
            torch.zeros(is_gt_p_on.sum(), 1).to(torch.float32).to(f_n.device),
            torch.zeros(is_gt_n_op.sum(), 1).to(torch.float32).to(f_n.device),
            torch.zeros(is_gt_n_on.sum(), 1).to(torch.float32).to(f_n.device),
        ),
        dim=0,
    ) # bsz_prime by 1

    Contrib_matrix_pos = torch.mm(Contrib_anchor, Contrib_contrast.T)
    Contrib_matrix_neg = 1.0 - Contrib_matrix_pos
    n_anchors = Contrib_anchor.shape[0]
    Contrib_matrix_pos[:, :n_anchors] -= torch.eye(n_anchors).to(f_n.device)

    return Output_anchor, Output_contrast, Contrib_matrix_pos, Contrib_matrix_neg, Conf_matrix


class WeightedBCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(WeightedBCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.weighted_bce = WeightedBCEWithLogitsLoss()        
        self.weighted_dice = WeightedDiceLoss()

    def forward(self, input, target, weight, pos_weight):
        # print("|DEBUG 4 - target.size() = {}".format(target.size()))
        # import pdb; pdb.set_trace()
        return self.alpha * self.weighted_bce(input, target, weight, pos_weight) + self.beta * self.weighted_dice(input, target, weight, pos_weight)

def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'WeightedBCEWithLogitsLoss':
        return WeightedBCEWithLogitsLoss()
    elif name == 'WeightedBCEDiceLoss':
        alpha = loss_config.get('alphs', 1.)
        beta = loss_config.get('beta', 1.)
        return WeightedBCEDiceLoss(alpha, beta)
    elif name == 'WeightedDiceLoss':
        normalization = loss_config.get('normalization', 'sigmoid')
        assert normalization == "sigmoid"
        return WeightedDiceLoss(normalization=normalization)
    elif name == 'CrossEntropyLoss':
        raise NotImplementedError()
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        raise NotImplementedError()
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        raise NotImplementedError()
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        raise NotImplementedError()
        normalization = loss_config.get('normalization', 'sigmoid')
        return GeneralizedDiceLoss(normalization=normalization)    
    elif name == 'MSELoss':
        raise NotImplementedError()
        return MSELoss()
    elif name == 'SmoothL1Loss':
        raise NotImplementedError()
        return SmoothL1Loss()
    elif name == 'L1Loss':
        raise NotImplementedError()
        return L1Loss()
    elif name == 'WeightedSmoothL1Loss':
        raise NotImplementedError()
        return WeightedSmoothL1Loss(threshold=loss_config['threshold'],
                                    initial_weight=loss_config['initial_weight'],
                                    apply_below_threshold=loss_config.get('apply_below_threshold', True))
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
