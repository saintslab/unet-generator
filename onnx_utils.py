import torch
import yaml 
import os
from model import UNet
from model_specs import UNetSpec
import onnx
import onnxruntime

def unet_constructor(loader, node):
    values = loader.construct_mapping(node)
    return UNetSpec(**values)

def spec_iterator(directory):
    if os.path.isdir(directory):
        files = list(map(lambda x: os.path.join(directory, x), os.listdir(directory)))
    else:
        files = [directory]

    for file in files:
        with open(file, 'r') as f:
            #yaml.add_constructor('!UNetSpec', unet_constructor)
            data = yaml.load(f, yaml.FullLoader)
            for model in data['models']:
                yield model

def torch_model_from_spec(spec):
    return UNet(UNetSpec(spec), initial_channels=spec['initial_channels'], final_channels=1)

def torch_to_onnx(torch_model, output_file, initial_channels):
    torch_model.eval()
    example_input = torch.randn(2, initial_channels, 512, 512)

    #torch.onnx.enable_fake_mode()
    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_program = torch.onnx.dynamo_export(torch_model, example_input, export_options=export_options)
    # onnx_program = torch.onnx.export(
    #          torch_model, 
    #          example_input, 
    #          output_file, 
    #          opset_version=18, 
    #          input_names=['input'],
    #          output_names=['output'],
    #          dynamic_axes={'input': {0: 'batch_size', 1: 'initial_channels'}, 'output': {0: 'batch_size'}}
    #      )
    onnx_program.save(output_file)
    #onnx.checker.check_model(onnx.load(output_file))    

def run_onnx(onnx_path, input_data):
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_data)}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs[0]

def compare_outputs(torch_model, onnx_path, initial_channels):
    torch_model.eval()
    input_data = torch.randn(10, initial_channels, 512, 512)
    torch_output = torch_model(input_data)
    onnx_output = run_onnx(onnx_path, input_data)
    torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=1e-3, atol=1e-3)