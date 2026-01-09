import torch
from util.case_util import get_torch_dtype, get_torch_device


class Case:
    def __init__(self):
        self.t_input: torch.Tensor = torch.tensor([])

    def save(self, path: str):
        tensors = {"input": self.t_input}
        torch.save(tensors, path)

    def load(self, path: str):
        tensors = torch.load(path)
        self.t_input = tensors["input"]

    def get_tensor_list(self) -> list:
        return [self.t_input]


class Result:
    def __init__(self):
        self.t_output: torch.Tensor = torch.tensor([])

    def save(self, path: str):
        tensors = {"output": self.t_output}
        torch.save(tensors, path)

    def load(self, path: str):
        tensors = torch.load(path)
        self.t_output = tensors["output"]

    def get_tensor_list(self) -> list:
        return [self.t_output]


def caller(case: Case, dtype, device) -> Result:
    torch_device = get_torch_device(device)
    torch_dtype = get_torch_dtype(dtype)
    if device == "baseline":
        torch_dtype = torch.float64

    result = Result()
    input_tensor = case.t_input.to(device=torch_device, dtype=torch_dtype)
    output_tensor = torch.nn.functional.softmax(input_tensor, dim=-1)
    result.t_output = output_tensor.to(device='cpu')
    return result