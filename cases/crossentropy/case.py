import torch
from util.case_util import get_torch_dtype, get_torch_device


class Case:
    def __init__(self):
        self.t_logits: torch.Tensor = torch.tensor([])
        self.t_targets: torch.Tensor = torch.tensor([])

    def save(self, path: str):
        tensors = {
            "logits": self.t_logits,
            "targets": self.t_targets,
        }
        torch.save(tensors, path)

    def load(self, path: str):
        tensors = torch.load(path)
        self.t_logits = tensors["logits"]
        self.t_targets = tensors["targets"]

    def get_tensor_list(self) -> list:
        return [self.t_logits, self.t_targets]


class Result:
    def __init__(self):
        self.t_loss: torch.Tensor = torch.tensor([])

    def save(self, path: str):
        tensors = {"loss": self.t_loss}
        torch.save(tensors, path)

    def load(self, path: str):
        tensors = torch.load(path)
        self.t_loss = tensors["loss"]

    def get_tensor_list(self) -> list:
        return [self.t_loss]


def caller(case: Case, dtype, device) -> Result:
    torch_device = get_torch_device(device)
    torch_dtype = get_torch_dtype(dtype)
    if device == "baseline":
        torch_dtype = torch.float64

    result = Result()
    logits_tensor = case.t_logits.to(device=torch_device, dtype=torch_dtype)
    targets_tensor = case.t_targets.to(device=torch_device, dtype=torch.long)
    
    # Handle 3D logits (batch, seq, vocab) -> reshape to (batch*seq, vocab)
    original_shape = logits_tensor.shape
    if logits_tensor.dim() == 3:
        batch_size, seq_len, vocab_size = logits_tensor.shape
        logits_tensor = logits_tensor.view(-1, vocab_size)
        targets_tensor = targets_tensor.view(-1)
    
    loss_tensor = torch.nn.functional.cross_entropy(logits_tensor, targets_tensor)
    result.t_loss = loss_tensor.to(device='cpu')
    return result