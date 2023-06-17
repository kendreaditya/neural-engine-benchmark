# T5/GPT-2 Model Conversion to Core ML

This script allows you to convert Hugging Face's T5/GPT-2 models to Core ML models. Core ML is a framework provided by Apple that enables deploying machine learning models on Apple devices.

## Xcode Core ML Performance Report
![](/assets/model-graph.png)

<div style='display: grid; grid-template-columns: repeat(2, 1fr); grid-gap: 10px;'>
  <div>
    <img src='./assets/gpt2-124m.png#1' alt= 'gpt2-124m'>
  </div>
  <div>
    <img src='./assets/gpt2-medium-355m.png#1' alt='gpt2-355m'>
  </div>
  <div>
    <img src='./assets/t5-base-220m.png#1' alt='t5-220m'>
  </div>
  <div>
    <img src='./assets/t5-small-60m.png#1' alt=;t5-60m'>
  </div>
</div>


## Prerequisites

- Python 3.10
- ane_transformers==0.1.1
- coremltools==6.3.0
- numpy==1.24.2
- tensorflow==2.12.0
- torch==2.0.0
- transformers==4.29.2
- transformers-exporter

`git clone https://github.com/huggingface/exporters.git && cd exporters && pip install -e .`

## Installation

1. Clone the repository:

`git clone https://github.com/kendreaditya/neural-engine-benchmark`

2. Install the required dependencies using pip:

   `pip install -r requirements.txt`

## Usage

To convert a T5/GPT-2 model to a Core ML model, follow these steps:

1. Open a terminal and navigate to the project directory.

2. Run the script with the desired T5/GPT model name as the argument.
   
   - Available T5 models are: `t5-small`, `t5-base`, `t5-large`, `t5-3b`, `t5-11b`
   - Available GPT-2 models are: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`

   For example, to convert the `t5-base` model, run:

   ```python t5-models.py t5-base```

1. The script will convert the T5 model to a Core ML model and save it as a `.mlmodel` file.

2. The converted Core ML model will be saved in the current directory with the following naming convention:

3. To verify performance, launch Xcode and simply add this model package file as a resource in their projects. From the Performance tab, you can generate a performance report on locally available devices, for example, on the Mac that is running Xcode or another Apple device that is connected to that Mac.

   `<model_name>-<num_params>.mlmodel`

For example, the `t5-base` model with 220 million parameters will be saved as `t5-base-220m.mlmodel`.

## Limitations

- This script is intended for converting T5 models to Core ML models only.
- Make sure you have enough disk space available as the converted models can be large in size.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This script is based on the work of [Apple Docs](https://coremltools.readme.io/docs/composite-operators), [Apple Blog](https://machinelearning.apple.com/research/neural-engine-transformers#figure3), [FastT5](https://github.com/madlag/fastT5) and [Hugging Face](https://huggingface.co/).
