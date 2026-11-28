"""Fixed-policy serving controls using the actual scheduler and model entry."""
import asyncio
import copy
import json
import math
import os
import time
from pathlib import Path


async def replay_prefix(agent, request_id, prompt_ids, output_ids, multimodal):
    if os.environ.get("KIMI_ALIGNMENT_SERVING_PROBE") != "1" or len(output_ids) < 5:
        return
    directory = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "serving_invariance"
    directory.mkdir(exist_ok=True)
    try:
        with (directory / "source.json").open("x") as handle:
            json.dump({"request_id": request_id, "prompt_ids": list(prompt_ids),
                       "suffix":list(output_ids[:4]),"target":int(output_ids[4])}, handle)
    except FileExistsError:
        return
    await agent.server_manager.generate(
        request_id="alignment-probe-control", prompt_ids=list(prompt_ids),
        sampling_params={"temperature": 0., "max_tokens": 1}, **multimodal)


def compare_checkpoint_replay(reference, replay):
    expected_ids, expected_scores = reference["ids"], reference["scores"]
    actual_ids, actual_scores = replay["token_ids"], replay["logprobs"]
    if not (len(expected_ids) == len(expected_scores) == len(actual_ids) == len(actual_scores) == 8):
        raise ValueError("Incomplete checkpoint replay")
    deltas = [abs(x - y) for x, y in zip(expected_scores, actual_scores, strict=True)]
    finite = all(math.isfinite(x) for x in expected_scores + actual_scores)
    first = next((i for i in range(len(deltas))
                  if actual_ids[i] != expected_ids[i] or deltas[i] != 0), None)
    return dict(valid_tokens=len(deltas), first_different_offset=first,
                differing_ids=sum(x != y for x,y in zip(expected_ids,actual_ids,strict=True)),
                differing_scores=sum(d != 0 for d in deltas),
                nonfinite=sum(not (math.isfinite(x) and math.isfinite(y))
                              for x,y in zip(expected_scores,actual_scores,strict=True)),
                mean_abs=sum(deltas)/len(deltas) if finite else None,
                max_abs=max(deltas) if finite else None, passed=finite and first is None)



def verify_reports(reports, directory):
    by_id = {r["id"].removeprefix("alignment-probe-"):r for r in reports}
    baseline = by_id["single-target"]
    checks = []
    def check(name, lhs, rhs, start=0):
        count = len(rhs["token_ids"])
        ids = lhs["token_ids"][start:start + count]
        scores = lhs["logprobs"][start:start + count]
        other = rhs["logprobs"]
        if not count or len(scores) != count or len(other) != count:
            raise ValueError(f"Incomplete serving replay {name}")
        finite = all(math.isfinite(x) for x in scores + other)
        delta = [abs(x-y) for x,y in zip(scores,other,strict=True)]
        checks.append(dict(case=name,valid_tokens=count,nonfinite=sum(
            not(math.isfinite(x) and math.isfinite(y)) for x,y in zip(scores,other,strict=True)),
            identical_history_and_tokens=ids==rhs["token_ids"],different=sum(x!=0 for x in delta),
            mean_abs=sum(delta)/count if finite else None,max_abs=max(delta) if finite else None,
            passed=finite and ids==rhs["token_ids"] and all(x==0 for x in delta)
                   and lhs['weight_version']==rhs['weight_version']))
    for label in ('repeat','mixed','reordered','batch16'):
        row=by_id[label+'-target']
        if len(row['token_ids'])!=len(baseline['token_ids']):
            raise ValueError(f'Incomplete length for {label}')
        check(label,baseline,row)
    check('decode-vs-full-prefix',baseline,by_id['full-prefix-target'],start=4)
    check('chunked-vs-full-prefill',by_id['long-full-target'],by_id['long-chunked-target'])
    (directory/'checks.json').write_text(json.dumps(checks,indent=2))
    if not all(row['passed'] for row in checks):
        raise RuntimeError(f"Multimodal BI/PD mismatch; see {directory}/checks.json")


async def before_first_sleep(server):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    directory = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "pre-sync"
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"replica-{server.replica_rank}.json"
    if dest.exists():
        return
    reference = json.loads(Path(os.environ["KIMI_ALIGNMENT_LONG_TRAJECTORY"]).with_name("single.json").read_text())[0]
    params = SamplingParams(temperature=0.,max_tokens=8,ignore_eos=True,logprobs=0,
                            output_kind=RequestOutputKind.FINAL_ONLY)
    stream = server.engine.generate({"prompt_token_ids":reference["prompt"]},params,
                                    f"initial-policy-text-{server.replica_rank}")
    async for output in stream:
        if output.finished:
            sample = output.outputs[0]
            replay = dict(token_ids=list(sample.token_ids), logprobs=[float(s[t].logprob) for t,s in zip(sample.token_ids,sample.logprobs,strict=True)])
            comparison = compare_checkpoint_replay(reference,replay)
            dest.write_text(json.dumps(dict(replica=server.replica_rank,replay=replay,comparison=comparison),indent=2))
    if not dest.exists():
        raise RuntimeError("Initial serving replay produced no final output")
    await server.engine.collective_rpc("capture_kimi_initial_state")


async def run_matrix(server, prompt):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    engine = server.engine
    directory = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "serving_invariance"
    await engine.wait_for_requests_to_drain()
    version = server.global_steps
    reports = []

    async def group(label, items, length, budget=None):
        await engine.pause_generation(mode="keep", clear_cache=False)
        previous_budget = None
        if budget is not None:
            previous_budget = await engine.engine_core.call_utility_async("alignment_scheduler_budget", budget)
        collectors = []
        ids = []
        try:
            for role, value in items:
                name = f"alignment-probe-{label}-{role}"
                params = SamplingParams(temperature=0., max_tokens=length,
                                        ignore_eos=True, logprobs=0,
                                        output_kind=RequestOutputKind.FINAL_ONLY)
                collector = await engine.add_request(name, value, params)
                collectors.append(collector)
                ids.append(name)
        finally:
            await engine.resume_generation()
        outputs = await asyncio.wait_for(asyncio.gather(*(c.get() for c in collectors)), float(os.getenv("KIMI_ALIGNMENT_REAL_MATRIX_TIMEOUT", "1800")))
        if previous_budget is not None:
            await engine.pause_generation(mode="keep", clear_cache=False)
            try:
                await engine.engine_core.call_utility_async("alignment_scheduler_budget", previous_budget)
            finally:
                await engine.resume_generation()
        for name, output in zip(ids, outputs):
            assert output.finished and len(output.outputs) == 1
            sample = output.outputs[0]
            reports.append({"id": name, "token_ids":list(sample.token_ids),
                            "logprobs":[float(scores[tok].logprob) for tok,scores in zip(sample.token_ids,sample.logprobs)],
                            "weight_version":version})
        (directory / "matrix-output.json").write_text(json.dumps(reports, indent=2))
        return outputs[0]

    # Same text trajectory as the independent, pre-PPO engines. Record this
    # before the longer multimodal checks so a failed reload can be diagnosed.
    reference_path = Path(os.environ["KIMI_ALIGNMENT_LONG_TRAJECTORY"]).with_name("single.json")
    reference = json.loads(reference_path.read_text())[0]
    await group("checkpoint-replay", [("target", {"prompt_token_ids":reference["prompt"]})], 8)
    comparison = compare_checkpoint_replay(reference, reports[-1])
    comparison.update(weight_version=version, prompt_tokens=len(reference["prompt"]),
                      reference=str(reference_path), server_pid=os.getpid())
    pre_sync = directory.parent / "pre-sync" / f"replica-{server.replica_rank}.json"
    if pre_sync.exists():
        before = json.loads(pre_sync.read_text())["replay"]
        same_server = compare_checkpoint_replay(dict(ids=before["token_ids"],scores=before["logprobs"]),reports[-1])
        same_server.update(replica=server.replica_rank,weight_version=version)
        (directory / "same-server-sync-replay.json").write_text(json.dumps(same_server,indent=2))
    (directory / "checkpoint-replay.json").write_text(json.dumps(comparison, indent=2))
    baseline = await group("single", [("target",prompt)], 64)
    await group("repeat", [("target",prompt)], 64)
    peer = copy.deepcopy(prompt)
    # Text-only suffix keeps the original multimodal placeholder geometry intact.
    peer["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + list(baseline.outputs[0].token_ids[:2])
    await group("mixed", [("target",prompt),("peer",peer)], 64)
    shorter_peer = copy.deepcopy(prompt)
    shorter_peer["prompt_token_ids"] = list(prompt["prompt_token_ids"])[:-2]
    await group("reordered", [("peer",shorter_peer),("target",prompt)], 64)
    neighbors = []
    for index in range(15):
        value = copy.deepcopy(prompt)
        value["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + list(baseline.outputs[0].token_ids[:index % 4 + 1])
        neighbors.append((f"peer{index}",value))
    await group("batch16", [("target",prompt), *neighbors], 64)
    full = copy.deepcopy(prompt)
    full["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + list(baseline.outputs[0].token_ids[:4])
    await group("full-prefix", [("target",full)], 1)
    extended = copy.deepcopy(prompt)
    extended["prompt_token_ids"] = list(prompt["prompt_token_ids"]) + list(baseline.outputs[0].token_ids[:8]) * 80
    await group("long-full", [("target",extended)], 1)
    await group("long-chunked", [("target",extended)], 1, budget=512)
    verify_reports(reports, directory)
    full_request = json.loads(Path(os.environ["KIMI_ALIGNMENT_LONG_TRAJECTORY"]).read_text())[0]
    if len(full_request["prompt"]) != 1024 or len(full_request["ids"]) != 2048:
        raise ValueError("Long response control lacks the required valid trajectory")
    long_checks = []
    for cut in (1023,1024,1025,2047):
        prefix = full_request["prompt"] + full_request["ids"][:cut-1]
        pair = await group(f"boundary-decode-{cut}", [("target",{"prompt_token_ids":prefix})],2)
        decoded = pair.outputs[0]
        first, second = decoded.token_ids
        one = await group(f"boundary-prefill-{cut}", [("target",{"prompt_token_ids":prefix+[first]})],1)
        replay = one.outputs[0]
        target = replay.token_ids[0]
        a = float(decoded.logprobs[1][second].logprob)
        b = float(replay.logprobs[0][target].logprob)
        long_checks.append(dict(prompt_length=len(prefix)+1,response_offset=cut,weight_version=version,
                               decode_id=second,prefill_id=target,decode_logprob=a,prefill_logprob=b,
                               abs_diff=abs(a-b),passed=second==target and math.isfinite(a) and a==b))
        (directory/"long-history-pd.json").write_text(json.dumps(long_checks,indent=2))
    if not all(x["passed"] for x in long_checks):
        raise RuntimeError("Same-policy long-history prefill/decode diverged")
    assert server.global_steps == version
    (directory / "matrix-complete.json").write_text(json.dumps({"version":version,"completed":True}))


def capture_probe_logits(logits, batch, rank):
    if rank != 0:
        return
    import torch
    directory = Path(os.environ["KIMI_ALIGNMENT_AUDIT_DIR"]) / "serving_invariance"
    for row, request in enumerate(batch.req_ids):
        if not request.startswith("alignment-probe-"):
            continue
        start,end = map(int,batch.query_start_loc[row:row+2].tolist())
        positions = batch.positions.flatten()[start:end]
        record = {"request":request,"positions":positions.cpu(),"batch_ids":list(batch.req_ids),
                  "ids":batch.input_ids.flatten()[start:end].cpu(),
                  "logits":logits[row].detach().cpu(),"total_queries":batch.input_ids.numel()}
        torch.save(record,directory / (request + f"-pos{int(positions[-1])}-{time.time_ns()}.pt"))
