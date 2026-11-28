"""Read-only sampled tensor comparison within the same vLLM process and TP rank."""
import hashlib
import json
import os
import socket
from pathlib import Path

import torch


@torch.no_grad()
def capture(worker, phase):
    if os.environ.get("KIMI_ALIGNMENT_SYNC_DIAGNOSTIC") != "1":
        return
    root = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "checkpoint-state"
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{socket.gethostname()}-{os.getpid()}"
    path = root / f"{prefix}-{phase}.json"
    if path.exists():
        return
    model = worker.model_runner.model
    selections = []
    index_cache = {}

    def sample(name, value):
        if value.numel() == 0 or value.layout != torch.strided or value.is_meta:
            return
        shape = tuple(value.shape)
        cache_key = (value.device, shape)
        if cache_key not in index_cache:
            count = min(128, value.numel())
            flat = torch.arange(count, device=value.device) * (value.numel()-1) // max(1,count-1)
            coords = []
            for size in reversed(shape):
                coords.append(flat % size)
                flat = flat // size
            index_cache[cache_key] = tuple(reversed(coords))
        picked = value.detach()[index_cache[cache_key]].reshape(-1).contiguous().view(torch.uint8)
        selections.append((name, list(shape), str(value.dtype), picked))

    for name, p in model.named_parameters():
        sample("param/"+name,p)
    for name, b in model.named_buffers():
        sample("buffer/"+name,b)
    for name, module in model.named_modules():
        for attr, val in vars(module).items():
            if isinstance(val, torch.Tensor) and not isinstance(val, torch.nn.Parameter):
                sample("attribute/"+name+"."+attr,val)
    records = {}
    for device in {v.device for _,_,_,v in selections}:
        group = [r for r in selections if r[3].device == device]
        flat = torch.cat([r[3] for r in group]).cpu().numpy()
        start = 0
        for name, shape, dtype, raw in group:
            end = start + raw.numel()
            payload = flat[start:end].tobytes()
            record = dict(shape=shape,dtype=dtype,sha256=hashlib.sha256(payload).hexdigest())
            if dtype == "torch.float32":
                values = torch.frombuffer(bytearray(payload),dtype=torch.float32)
                rounded = values.to(torch.bfloat16)
                record["bf16_roundtrip_sha256"] = hashlib.sha256(rounded.float().numpy().tobytes()).hexdigest()
                record["bf16_sha256"] = hashlib.sha256(rounded.view(torch.uint8).numpy().tobytes()).hexdigest()
            records[name] = record
            start = end
    path.write_text(json.dumps(dict(pid=os.getpid(),node=socket.gethostname(),local_rank=worker.local_rank,items=records)))
    if phase == "after":
        before = json.loads((root / f"{prefix}-before.json").read_text())["items"]
        changed = [k for k in sorted(records.keys() & before.keys()) if records[k] != before[k]]
        result = dict(total=len(records),changed=changed,missing=sorted(before.keys()-records.keys()),new=sorted(records.keys()-before.keys()))
        rounded = []
        for name in changed:
            source, dest = before[name], records[name]
            key = "bf16_roundtrip_sha256" if dest["dtype"]=="torch.float32" else "bf16_sha256" if dest["dtype"]=="torch.bfloat16" else None
            if key and source.get(key)==dest["sha256"] and source["shape"]==dest["shape"]:
                rounded.append(name)
        result["changed_matching_bf16_rounding"] = rounded
        (root / f"{prefix}-comparison.json").write_text(json.dumps(result))
