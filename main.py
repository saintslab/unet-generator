import argparse
import os
from model_specs import *
from model import *
from tqdm import tqdm
import yaml
from onnx_test import minitest
from onnx_utils import run_onnx, spec_iterator, torch_model_from_spec, torch_to_onnx, compare_outputs


def write_batch(dir, batch, total_models, batch_idx):
    current_dir = os.path.curdir
    with open(os.path.join(current_dir, dir, f'model_batch_{batch_idx}.yaml'), 'w') as f:
        f.write(yaml.dump({'model_spec_version': MODEL_SPEC_VERSION,
                'batch_id': batch_idx, 'total_models': total_models, 'models': batch}))


def generate(args):
    if args.output is None or not os.path.isdir(args.output):
        print('Error: Output directory does not exist')
    if args.exhaustive:
        print('Generating exhaustive model specifications..')
        l = combinations_length()
        batch = []
        batch_idx = 0
        for model in tqdm(model_iterator(), total=l):
            batch.append(model)
            if len(batch) == 1000:
                write_batch(args.output, batch, l, batch_idx)
                batch_idx += 1
                batch = []
        if len(batch) > 0:
            write_batch(args.output, batch, l, batch_idx)
    if args.n is not None:
        print('Generating random model specifications..')
        batch = sample(args.n)
        write_batch(args.output, batch, args.n, 0)

def test(args):
    #spec = sample(1)[0]
    spec = UNetSpec()
    spec.depth = 1
    spec.kernel_sizes = [3,1]
    spec.initial_channels = 1
    model = UNet(spec, initial_channels=spec.initial_channels, final_channels=1)
    test_data = torch.randn(2, spec.initial_channels, 512, 512)
    print(model.forward(test_data))

def onnx(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    if not os.path.isdir(input_dir) and not os.path.isfile(input_dir):
        print('Error: Input file/directory does not exist')
        raise FileNotFoundError

    if not os.path.isdir(output_dir):
        print('Error: Output directory does not exist')
        raise FileNotFoundError
    
    for idx, spec in tqdm(enumerate(spec_iterator(input_dir))):
        model = torch_model_from_spec(spec)
        print(len(list(model.parameters())))
        torch_to_onnx(model, os.path.join(output_dir, f'model_{idx}.onnx'), spec['initial_channels'])

def sample_models(args):
    n = args.n
    output_file = args.output_file
    if not '.yaml' in output_file:
        print('Error: Output file has to end with .yaml')
        raise FileNotFoundError()
    
    if n > combinations_length():
        print(f'Error: n is greater than the number of possible model specifications (n > {combinations_length()})')
        raise Exception()
    
    models = ancestral_sampling_without_replacement(n)

    yaml.add_representer(UNetSpec, lambda dumper, data: dumper.represent_dict(data.__dict__))
    with open(output_file, 'w') as f:
        f.write(yaml.dump({'model_spec_version': MODEL_SPEC_VERSION,
            'total_models': len(models), 'models': models}))

# REMOVED since it didn't make sense (generating torch model from spec samples new parameters, which cannot be compared with ONNX)
# def test_onnx(args):
#     sample_file = args.sample_file
#     onnx_dir = args.onnx_directory
#     if not os.path.isdir(onnx_dir):
#         print('Error: ONNX directory does not exist')
#         raise FileNotFoundError
    
#     if not os.path.isfile(sample_file) and not os.path.isdir(sample_file):
#         print('Error: Sample file/directory does not exist')
#         raise FileNotFoundError
    
#     with open(sample_file, 'r') as f:
#         data = yaml.load(f, yaml.FullLoader)
#         for idx, model in tqdm(enumerate(data['models'])):
#             torch_model = torch_model_from_spec(model)
#             compare_outputs(torch_model, os.path.join(onnx_dir, f'model_{idx}.onnx'), model['initial_channels'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    generate_parser = subparsers.add_parser(
        'generate', help='Generate model specifications (yaml)')
    generate_parser.add_argument(
        '-n', type=int, help='Amount of model specifications to generate')
    generate_parser.add_argument(
        '-o', '--output', type=str, help='Name of output file', required=True)
    generate_parser.add_argument('--exhaustive', type=bool, default=False,
                                 help='Generates all possible specifications, ignores -n. WARNING: May be large number.')
    generate_parser.set_defaults(func=generate)

    test_parser = subparsers.add_parser('test', help='Test model generation')
    test_parser.set_defaults(func=test)

    onnx_parser = subparsers.add_parser('onnx', help='Generate ONNX models from model spec directory')
    onnx_parser.add_argument('input_dir', type=str, help='Directory containing model specifications')
    onnx_parser.add_argument('output_dir', type=str, help='Directory to contain ONNX model files')
    onnx_parser.set_defaults(func=onnx)

    sample_parser = subparsers.add_parser('sample', help='Sample model specifications without replacement')
    sample_parser.add_argument('output_file', type=str, help='File to write model specifications to')
    sample_parser.add_argument('-n', type=int, help='Amount of samples to generate')
    sample_parser.set_defaults(func=sample_models)

    # test_onnx_parser = subparsers.add_parser('test_onnx', help='Test ONNX model generation')
    # test_onnx_parser.add_argument('sample_file', type=str, help='YAML file/directory with samples')
    # test_onnx_parser.add_argument('onnx_directory', type=str, help='Directory with ONNX models')
    # test_onnx_parser.set_defaults(func=test_onnx)

    minitest_parser = subparsers.add_parser('minitest', help='Test simple ONNX model vs Torch model')
    minitest_parser.add_argument('output', type=str, help='ONNX output file')
    minitest_parser.set_defaults(func=minitest)

    args = parser.parse_args()
    args.func(args)
