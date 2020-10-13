from thop import profile
from Network.igcV1_unet import IGCV1_UNet
import torch

input_size = torch.randn((1, 1, 1, 192, 192))
net = IGCV1_UNet()

flops, params = profile(model=net, inputs=input_size)

print(flops)
print(params)

