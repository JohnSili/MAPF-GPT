# from pathlib import Path
# from typing import Literal, Optional
#
# import cppimport.import_hook
# import torch
# from huggingface_hub import hf_hub_download
# from pogema_toolbox.algorithm_config import AlgoBase
# from pogema_toolbox.registry import ToolboxRegistry
# from pydantic import Extra
#
# from mapf_gpt.model import GPT, GPTConfig
# from tokenizer import cost2go
# from tokenizer.tokenizer import Encoder, InputParameters

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Optional

import cppimport.import_hook
import torch
from huggingface_hub import hf_hub_download
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import Extra

import torch.nn as nn

from mapf_gpt.model import GPT, GPTConfig
from mapf_gpt.observation_generator import ObservationGenerator, InputParameters

try:
    from torch.ao.quantization import quantize_dynamic
except ImportError:
    from torch.quantization import quantize_dynamic


def _break_wte_lm_head_tie(model: GPT) -> None:
    """Embedding and lm_head share weights in GPT; int8 quantization needs separate tensors."""
    w = model.transformer.wte.weight.data.clone()
    model.transformer.wte.weight = nn.Parameter(w.clone())
    model.lm_head.weight = nn.Parameter(w.clone())


class MAPFGPTInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["MAPF-GPT"] = "MAPF-GPT"
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    path_to_weights: Optional[str] = "weights/model-6M.pt"
    device: str = "cuda"
    # float32 | bfloat16 | float16 — for non-quantized weights only (GPU recommended)
    infer_dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    repo_id: str = 'aandreychuk/MAPF-GPT'
    grid_step: int = 64
    save_cost2go: bool = False
    batch_size: int = 2048
    num_process: int = 8

def strip_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    """
    Strips DDP prefix from keys. Preserves OrderedDict and _metadata (required for
    torch quantized modules when loading from disk).
    """
    keys = list(state_dict.keys())
    if not any(k.startswith(prefix) for k in keys):
        return copy.copy(state_dict)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix) :]] = v
        else:
            new_state_dict[k] = v
    if getattr(state_dict, "_metadata", None):
        new_meta = OrderedDict()
        for mk, mv in state_dict._metadata.items():
            if mk.startswith(prefix):
                new_meta[mk[len(prefix) :]] = mv
            else:
                new_meta[mk] = mv
        new_state_dict._metadata = new_meta
    return new_state_dict


class MAPFGPTInference:
    def __init__(self, cfg: MAPFGPTInferenceConfig, net=None):
        self.cfg: MAPFGPTInferenceConfig = cfg
        self.input_parameters = InputParameters(
            cfg.cost2go_value_limit,
            cfg.num_agents,
            cfg.num_previous_actions,
            cfg.context_size,
            cfg.cost2go_radius,
            cfg.agents_radius,
            cfg.grid_step,
            cfg.save_cost2go
        )
        self.observation_generator = None
        self.last_actions = None

        path_to_weights = Path(self.cfg.path_to_weights)
        if path_to_weights.name in ['model-2M.pt', 'model-6M.pt', 'model-85M.pt']:
            hf_hub_download(repo_id=self.cfg.repo_id, filename=path_to_weights.name, local_dir=path_to_weights.parent)
            ToolboxRegistry.info(f'Using weights loaded from huggingface: {path_to_weights}')

        requested_device = self.cfg.device
        checkpoint = torch.load(Path(self.cfg.path_to_weights), map_location="cpu")

        if checkpoint.get("quantized"):
            if requested_device != "cpu":
                ToolboxRegistry.warning(
                    "Quantized checkpoint (dynamic int8) runs on CPU in PyTorch; "
                    f"ignoring device {requested_device!r}."
                )
            self.cfg.device = "cpu"
        elif ('cuda' in self.cfg.device and not torch.cuda.is_available()) or (
            self.cfg.device == 'mps' and not torch.backends.mps.is_available()
        ):
            ToolboxRegistry.warning(f'{self.cfg.device} is not available, using cpu instead!')
            self.cfg.device = 'cpu'

        self.torch_generator = torch.Generator(device=self.cfg.device)
        self.torch_generator.manual_seed(0)

        model_state_dict = strip_prefix_from_state_dict(checkpoint["model"])
        config_dict = checkpoint.get("model_args")
        gpt_config = GPTConfig(**config_dict)
        if net is not None:
            self.net = net
        else:
            if checkpoint.get("quantized"):
                self.net = GPT(gpt_config)
                _break_wte_lm_head_tie(self.net)
                self.net = quantize_dynamic(self.net, {nn.Linear}, dtype=torch.qint8)
                self.net.load_state_dict(model_state_dict, strict=False)
                self.net.eval()
            else:
                self.net = GPT(gpt_config)
                self.net.load_state_dict(model_state_dict, strict=False)
                self.net.to(self.cfg.device)
                self.net.eval()
                self._apply_infer_dtype()

    def _apply_infer_dtype(self) -> None:
        name = self.cfg.infer_dtype
        if name == "float32":
            return
        dt = getattr(torch, name)
        if name == "bfloat16" and "cuda" in self.cfg.device and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                ToolboxRegistry.warning(
                    "BF16 is not supported on this CUDA device; keeping float32."
                )
                return
        self.net = self.net.to(dtype=dt)

    def act(self, observations):
        if isinstance(observations[0], dict):
            positions = [obs["global_xy"] for obs in observations]
            goals = [obs["global_target_xy"] for obs in observations]
            if self.observation_generator is None:
                self.observation_generator = ObservationGenerator(observations[0]["global_obstacles"].copy().astype(int).tolist(), self.input_parameters)
                self.observation_generator.create_agents(positions, goals)
                self.last_actions = [-1 for _ in range(len(observations))]
            self.observation_generator.update_agents(positions, goals, self.last_actions)
            inputs = self.observation_generator.generate_observations()
        else:
            inputs = observations
        if len(inputs) > self.cfg.batch_size:
            actions = []
            for i in range(0, len(inputs), self.cfg.batch_size):
                batch_inputs = inputs[i:i + self.cfg.batch_size]
                tensor_obs = torch.tensor(batch_inputs, dtype=torch.long, device=self.cfg.device)
                batch_actions = torch.squeeze(self.net.act(tensor_obs, generator=self.torch_generator)).tolist()
                actions.extend(batch_actions)
        else:
            tensor_obs = torch.tensor(inputs, dtype=torch.long, device=self.cfg.device)
            actions = torch.squeeze(self.net.act(tensor_obs, generator=self.torch_generator)).tolist()
        if not isinstance(actions, list):
            actions = [actions]
        self.last_actions = actions.copy()
        return actions

    def reset_states(self):
        self.observation_generator = None
        self.torch_generator.manual_seed(0)
