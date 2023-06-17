import os
import argparse
import coremltools as ct
from coremltools.converters.mil.frontend.tensorflow.tf_op_registry import _TF_OPS_REGISTRY
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_tf_op
from transformers import TFT5Model
from transformers import AutoTokenizer

model_list = {
    't5-small': '60m',
    't5-base': '220m',
    't5-large': '770m',
    't5-3b': '3000m',
    't5-11b': '11000m'
}

def t5_mlmodel(name):
    model = TFT5Model.from_pretrained(name)

    # Disable the unsupported Einsum operation
    del _TF_OPS_REGISTRY["Einsum"]

    # Define a composite function for Einsum operation using MIL operators
    @register_tf_op
    def Einsum(context, node):
        assert node.attr['equation'] == 'bnqd,bnkd->bnqk'
        a = context[node.inputs[0]]
        b = context[node.inputs[1]]
        x = mb.matmul(x=a, y=b, transpose_x=False, transpose_y=True, name=node.name)
        context.add(node.name, x)

    # Convert the model using Core ML converter
    mlmodel = ct.convert(model, source="tensorflow")
    return mlmodel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face's T5 models to Core ML models")
    parser.add_argument("model_name", choices=model_list.keys(), help="Name of the T5 model")
    args = parser.parse_args()

    model_name = args.model_name
    num_params = model_list[model_name]

    mlmodel = t5_mlmodel(model_name)
    mlmodel.save(f"{model_name}-{num_params}.mlmodel")