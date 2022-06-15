from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.models.resnet import ResNet, Bottleneck
from timm.models.vision_transformer import VisionTransformer

from src.config import Config


class ImageClsModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(ImageClsModel, self).__init__()

        if config.model_type == 'resnext':
            self.model: nn.Module = ResNet(Bottleneck, [3, 3, 3, 0], num_classes=100,
                                           cardinality=32, base_width=4,
                                           drop_path_rate=min(0.99, config.drop_path_rate))

            self.model.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
            self.model.fc = nn.Linear(1024, 100)
        elif config.model_type == 'vit':
            self.model = VisionTransformer(img_size=32, patch_size=2, num_classes=100,
                                           embed_dim=192, depth=11, num_heads=3,
                                           drop_path_rate=min(0.99, config.drop_path_rate))
        else:
            self.model = VisionTransformer(img_size=32, patch_size=2, num_classes=100,
                                           embed_dim=384, depth=4, num_heads=6, mlp_ratio=2,
                                           drop_path_rate=min(0.99, config.drop_path_rate))

        self.train_ce = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.eval_ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(
        self, x: torch.Tensor, target: Optional[torch.Tensor] = None, pred_eval: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        raw_output: torch.Tensor = self.model(x)

        if target is None:
            return raw_output.argmax(1) if pred_eval else raw_output
        elif not pred_eval:
            return self.train_ce(raw_output, target)
        else:
            match = raw_output.topk(5).indices == target.view(-1, 1)
            return self.eval_ce(raw_output, target), match[:, 0].sum(), match.sum()
