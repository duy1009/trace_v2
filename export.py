import torch
import sys,os
sys.path.append(os.path.dirname(__file__))
from model import TraceModel
from collections import OrderedDict

MODEL_PATH = r"/home/hungdv/tcgroup/Clone_project/trace/weights/trace_wtw.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
net = TraceModel()
net.to(device=device)
state_dict = torch.load(MODEL_PATH, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(copyStateDict(state_dict))
MODE = 0


torch_input = torch.randn(1, 3, 640, 640)
print(os.path.basename(MODEL_PATH).split(".")[0] + ".onnx")


onnx_program = torch.onnx.export(net, 
                                 torch_input, 
                                 os.path.basename(MODEL_PATH).split(".")[0] + ".onnx",
                                 dynamic_axes={'input' : {0 : 'batch_size'}, 
                                               'output' : {0 : 'batch_size'}}
                                 )
