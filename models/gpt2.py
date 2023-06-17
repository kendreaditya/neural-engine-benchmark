import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import coremltools as ct
import argparse


class FinishMySentence(torch.nn.Module):
    def __init__(self, model=None, eos=198):
        super(FinishMySentence, self).__init__()
        self.eos = torch.tensor([eos])
        self.next_token_predictor = model
        self.default_token = torch.tensor([0])

    def forward(self, x):
        sentence = x
        token = self.default_token
        while token != self.eos:
            predictions, _ = self.next_token_predictor(sentence)
            token = torch.argmax(predictions[-1, :], dim=0, keepdim=True)
            sentence = torch.cat((sentence, token), 0)

        return sentence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to Core ML format")
    parser.add_argument("model_name", type=str, help="Name of the pretrained model")
    args = parser.parse_args()
    model_name = args.model_name

    token_predictor = GPT2LMHeadModel.from_pretrained(model_name, torchscript=True).eval()

    random_tokens = torch.randint(10000, (5,))
    traced_token_predictor = torch.jit.trace(token_predictor, random_tokens)

    model = FinishMySentence(model=traced_token_predictor)
    scripted_model = torch.jit.script(model)

    mlmodel = ct.convert(
        scripted_model,
        inputs=[ct.TensorType(name="context", shape=(ct.RangeDim(1, 64),), dtype=np.int32)],
    )

    mlmodel.save(f"{model_name}.mlmodel")