# U-Net generator script

This repository contains a script to enumerate/sample U-Net topologies and ONNX models from the space defined as so:

$$\textrm{depth} = \\{ 1,2,3,4 \\}$$

$$\textrm{kernel\\_size} = \\{1,3,5\\}$$

$$\textrm{U-net} = \\{\textrm{kernel\\_size}^{d + 1}\ |\ d \in \textrm{depth}\\}$$

The reason for $d+1$ instead of $d$ is due to the middle connecting layer, which is also present in the original U-Net paper.
This space spans 360 different topologies, but it is easy to expand it to more.

## Installation
Requires Python 3.11. Only tested on Fedora Linux, but should work on other platforms as well.

To install, run `pip install -r requirements.txt` or similar in your favorite virtual environment.

## Usage
**Print help message**
`python main.py --help`

**Sample $n$ models from the space**
~~~bash
python main.py sample <file (.yaml)> -n=<int>
~~~

Samples `n` topology specifications (depth, kernel sizes) without replacement and saves them in a `.yaml` file. This is only a small amount of metadata, so it doesn't require much time nor space.


**Compile ONNX models from topology specifications**

~~~bash
python main.py onnx <file.yaml> <output_dir>
~~~
Requires `output_dir` to exist. For each topology specified in `file.yaml`, compiles an ONNX model and places it in `output_dir` with the name `model_{id}.onnx` where `id` is its index in `file.yaml`.

**Generate all possible topologies**
~~~bash
python main.py generate -o <output_dir> --exhaustive true
~~~

Will generate all possible topologies and place specifications in folder `<output_dir>`. May create one or more files, depending on the amount of topologies.
