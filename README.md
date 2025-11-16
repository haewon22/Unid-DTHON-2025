# Unid-DTHON-2025

1. trian.py가 실행되지 않는다면, model_train.py로 실행
2. 경로 수정
3. stage1 -> stage2로 넘어가는 과정에서 bn error 발생 시,
(1)   코드 상단에 아래 코드 첨부

```
import torch.nn.functional as F
import doclayout_yolo.nn.modules.g2l_crm as g2l_crm

def _patched_dilated_conv(self, x, dilation):
    weight = self.dcv.conv.weight
    padding = dilation * (self.k // 2)

    x = F.conv2d(x, weight, stride=1, padding=padding, dilation=dilation)

    if hasattr(self.dcv, "bn") and self.dcv.bn is not None:
        x = self.dcv.bn(x)
    if hasattr(self.dcv, "act") and self.dcv.act is not None:
        x = self.dcv.act(x)
    return x

g2l_crm.DilatedBlock.dilated_conv = _patched_dilated_conv
```

(2) 또는 cmd에서   
```python train_doclayout.py —gpu_id 0 —sample_ratio 0.1 —epochs 5 —batch_size 32  —skip_dla``` 입력   


* data preprocessing은 제출 코드에 포함되어있습니다. 
