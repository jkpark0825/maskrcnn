import torch
from torch import nn, Tensor

from torch.nn.modules.utils import _pair
from torch.jit.annotations import BroadcastingList2

from torchvision.extension import _assert_has_ops
from .utils import convert_boxes_to_roi_format


def roi_align(
    input: Tensor,
    boxes: Tensor,
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> Tensor:
    
    _assert_has_ops()
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    return torch.ops.torchvision.roi_align(input, rois, spatial_scale,
                                           output_size[0], output_size[1],
                                           sampling_ratio, aligned)


class RoIAlign(nn.Module):
    def __init__(
        self,
        output_size: BroadcastingList2[int],
        spatial_scale: float,
        sampling_ratio: int,
        aligned: bool = False,
    ):
        super(RoIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)

