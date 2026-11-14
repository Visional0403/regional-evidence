from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


class _IdentityTransform(nn.Module):

    def forward(self, images: Sequence[torch.Tensor], targets: Any=None):
        from torchvision.models.detection.image_list import ImageList
        if len(images) != 1:
            raise ValueError('Physical batch size is fixed to one; use gradient accumulation.')
        sizes = [tuple(map(int, image.shape[-2:])) for image in images]
        return (ImageList(torch.stack(list(images)), sizes), targets)

    def postprocess(self, result: Any, _image_sizes: Any, _original_sizes: Any) -> Any:
        return result

def make_retinanet(backbone: nn.Module, *, num_classes: int=11) -> nn.Module:
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import boxes as box_ops
    anchors = AnchorGenerator(sizes=((8,), (16,), (32,), (64,), (128,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    class IgnoreAwareRetinaNet(RetinaNet):

        def compute_loss(self, targets, head_outputs, generated_anchors):
            matched = []
            for anchors_per_image, target in zip(generated_anchors, targets, strict=True):
                if target['boxes'].numel():
                    quality = box_ops.box_iou(target['boxes'], anchors_per_image)
                    indices = self.proposal_matcher(quality)
                else:
                    indices = torch.full((anchors_per_image.shape[0],), -1, dtype=torch.int64, device=anchors_per_image.device)
                ignored = target.get('ignored_boxes')
                if ignored is not None and ignored.numel():
                    ignored_iou = box_ops.box_iou(ignored, anchors_per_image).amax(dim=0)
                    indices = indices.masked_fill(ignored_iou >= 0.5, -2)
                matched.append(indices)
            return self.head.compute_loss(targets, head_outputs, generated_anchors, matched)
    model = IgnoreAwareRetinaNet(backbone, num_classes=num_classes, anchor_generator=anchors)
    model.transform = _IdentityTransform()
    return model
