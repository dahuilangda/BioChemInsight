import os
import random
import numpy as np
import torch
import torch.distributed as dist
import math
import time
import datetime
import json

def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def save_args(args):
    dt = datetime.datetime.strftime(datetime.datetime.now(), "%y%m%d-%H%M")
    path = os.path.join(args.log_dir, f'args_{dt}.txt')
    with open(path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"**** {k} = *{v}*\n")
    return


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if type(data) is list:
        return [to_device(v, device) for v in data]
    if type(data) is dict:
        return {k: to_device(v, device) for k, v in data.items()}


def round_floats(o):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def format_df(df):
    def _dumps(obj):
        if obj is None:
            return obj
        return json.dumps(round_floats(obj)).replace(" ", "")
    for field in ['node_coords', 'node_symbols', 'edges']:
        if field in df.columns:
            df[field] = [_dumps(obj) for obj in df[field]]
    return df

def adapt_state_dict(lora_model, old_state_dict, verbose=True):
    """
    通用加载器：同时兼容
    - Linear-only 模型权重（没有 .base）
    - LoRA 模型权重（含 .base 和 lora_A/B）
    会尽量优先加载完整LoRA权重，如果缺失则回退到Linear-only权重。
    """
    new_state_dict = {}
    model_keys = dict(lora_model.state_dict().items())
    for target_k in model_keys.keys():
        # 如果目标key是LoRALinear的base分支，比如 attn.q_proj.base.weight
        if ".base." in target_k:
            base_k = target_k
            linear_k = target_k.replace(".base.", ".")  # 回退到老Linear-only key
            if base_k in old_state_dict:
                val = old_state_dict[base_k]
                if verbose:
                    print(f"[LoRA-load] Found {base_k} -> {target_k}")
            elif linear_k in old_state_dict:
                val = old_state_dict[linear_k]
                if verbose:
                    print(f"[Linear-load] Found {linear_k} -> {target_k}")
            else:
                if verbose:
                    print(f"[Missed] No match for {target_k}, keep current")
                continue
            new_state_dict[target_k] = val
        else:
            # 对于非LoRA参数，直接照搬
            if target_k in old_state_dict:
                new_state_dict[target_k] = old_state_dict[target_k]
                if verbose:
                    print(f"[Direct-load] Found {target_k}")
            else:
                #new_state_dict[target_k] = model_keys[target_k]
                if verbose:
                    print(f"[Missed] No match for {target_k}, keep current")
                continue
    
    return new_state_dict
