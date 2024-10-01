import torch

from model import UNet
from model_specs import UNetSpec
from onnx_utils import run_onnx

# class MiniModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.relu = torch.nn.ReLU()
#         self.linear = torch.nn.Linear(1,1)

#     def forward(self, x):
#         x = self.relu(self.linear(x))
#         return x

def minitest(args):
    spec = UNetSpec()
    spec.depth = 2
    spec.kernel_sizes = [3,3, 3]
    spec.initial_channels = 10

    torch_model = UNet(spec, initial_channels=10)
    torch_model.eval()
    #export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    x = torch.randn(10,10,512,512)
    #onnx_model = torch.onnx.dynamo_export(torch_model, torch.randn(10,10,512,512), export_options=export_options)
    #onnx_model.save(args.output)
    torch.onnx.export(
            torch_model, 
            (x,), 
            args.output,
            opset_version=18, 
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 1: 'initial_channels'}, 'output': {0: 'batch_size'}}
    )
    torch.testing.assert_close(torch_model(x), torch.tensor(run_onnx(args.output, x)), rtol=1e-3, atol=1e-3)