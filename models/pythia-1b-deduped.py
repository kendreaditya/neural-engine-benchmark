# %%
import torch
import numpy as np
from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPT2LMHeadModel
import coremltools as ct
import argparse

from transformers.utils import (
    TensorType,
)
# %%
model_name = "EleutherAI/pythia-1b-deduped"
model = GPTNeoXForCausalLM.from_pretrained(model_name, torchscript=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
input = tokenizer("Hello, I am", return_tensors="pt")

# %%
from exporters import coreml
preprocessor = tokenizer
# Create dummy input data for doing the JIT trace.
from exporters.coreml.features import FeaturesManager
from exporters.coreml.convert import Wrapper 
from exporters.coreml.config import CoreMLConfig

model_kind, model_coreml_config= FeaturesManager.check_supported_model_or_raise(model, feature="feature-extraction")
coreml_config = model_coreml_config(model.config, use_past=False, seq2seq=None)
dummy_inputs = coreml_config.generate_dummy_inputs(preprocessor, framework=TensorType.PYTORCH)

# Put the inputs in the order from the config.
model_kind, config = FeaturesManager.check_supported_model_or_raise(model, feature="feature-extraction")
example_input = [dummy_inputs[key][0] for key in list(coreml_config.inputs.keys())]

wrapper = Wrapper(preprocessor, model, coreml_config).eval()

# Running the model once with gradients disabled prevents an error during JIT tracing
# that happens with certain models such as LeViT. The error message is: "Cannot insert
# a Tensor that requires grad as a constant."

with torch.no_grad():
    dummy_output = wrapper(*example_input)

traced_model = torch.jit.trace(wrapper, example_input, strict=True)

# %%
scripted_model = torch.jit.script(traced_model)

# %%
from exporters.coreml.convert import get_input_types

input_tensors = get_input_types(preprocessor, coreml_config, dummy_inputs)


# %%
mlmodel = ct.convert(
    scripted_model,
    inputs=input_tensors,
)

# %%
input_shape = ct.EnumeratedShapes(
        shapes=[[1, 1, 2048, 64],
                [2, 1, 2048, 64]],
        default=[1, 1, 2048, 64]
    )

mlmodel = ct.convert(
    scripted_model,
    inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=input_shape, dtype=np.int32)]
)
# %%
# Just model with no Wrapper

traced_model_only = torch.jit.trace(model, example_input, strict=True)

mlmodel = ct.convert(
    traced_model_only,
    inputs=input_tensors,
)
# %%
mlmodel = ct.convert(
    scripted_model,
    inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1, upper_bound=128),), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(lower_bound=1, upper_bound=128),), dtype=np.int32)]
    # outputs=[ct.TensorType(name="output_0")]
)
# %%
mlmodel.save(f"../mlmodels/{model_name}.mlmodel")
# %%
