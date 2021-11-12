import io
import torch
import torch.onnx
from reactnet import reactnet

model = reactnet()

pthfile = './checkpoint.pth'
state_dict = torch.load(pthfile, map_location='cpu')['state_dict']

# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove module prefix.
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)

#data type nchw
dummy_input1 = torch.randn(1, 3, 224, 224)
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, dummy_input1, "reactnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
