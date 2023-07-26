# %%
import coremltools as ct
from coremltools.models.neural_network import quantization_utils

# %%
# load full precision model
model_fp32 = ct.models.MLModel('/Users/kendreaditya/Downloads/weights/gpt2-medium/Model.mlpackage')

# %%
model_fp4 = quantization_utils.quantize_weights(model_fp32, nbits=4)
# %%
model_fp4.save(f"quant4.mlmodel")
# %%
