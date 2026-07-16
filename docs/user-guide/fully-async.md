---
title: Fully Async Rollout
description: How fully async rollout decouples generation from training, when to use it, and which flags enable it.
---
Fully async rollout splits Miles into two concurrent loops:

1. A background rollout worker keeps SGLang generation in flight and pushes completed
   samples into a queue.
2. The trainer drains the queue, runs optimizer steps, and syncs updated weights back
   to rollout engines.

When rollout and training take similar time, per-iteration wall time moves from
`rollout_time + train_time` toward `max(rollout_time, train_time)`.

## When to use it

| Use fully async when | Stay synchronous when |
|---|---|
| Rollout is a large part of wall time | Debugging a new recipe |
| The run is long enough to amortize queue warm-up | You need the strictest possible on-policy cadence |
| SGLang engines can keep many requests in flight | Queue depth stays at zero even after tuning concurrency |
| You can tolerate slightly older samples in exchange for throughput | You are validating loss math or reward plumbing |

The mode is especially useful for long-context math, tool-use, and agentic workloads
where generation dominates the iteration.

## Enable it

Switch the entrypoint from `train.py` to `train_async.py` and provide a rollout
function that owns the background worker:

```diff
- python3 train.py ...
+ python3 train_async.py ...
+   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
```

Everything else belongs in the same [argument groups](/user-guide/argument-groups) as a
synchronous run.

## Queue model

```mermaid
sequenceDiagram
    participant T as Trainer
    participant Q as Rollout queue
    participant W as Async worker
    participant S as SGLang engines

    par Producer
        loop continuously
            W->>S: generate(prompt)
            S-->>W: response
            W->>Q: enqueue sample
        end
    and Consumer
        loop each trainer iteration
            T->>Q: drain batch
            T->>T: optimizer step
            T->>S: sync weights
        end
    end
```

The queue is the contract. If it stays populated, the trainer does not wait for
generation. If it is empty, rollout is still the bottleneck and async cannot hide it.

## Tuning knobs

| Knob | What it changes |
|---|---|
| `--rollout-batch-size` | Target amount of work the async producer keeps in flight |
| `--sglang-server-concurrency` | Per-engine request concurrency |
| `--global-batch-size` | Number of samples the trainer drains per step |
| `--num-steps-per-rollout` | Number of optimizer steps per queue drain cycle. Values `> 1` reintroduce intra-drain off-policyness on top of the rollout-vs-trainer gap; see [Off-policyness and steps per drain](#off-policyness-and-steps-per-drain) |
| `--max-weight-staleness` | When the rollout engine's weight version lags the trainer's by more than this, the worker recycles the stale group instead of feeding it to the loss |
| `--use-dynamic-global-batch-size` | Resize global batch to the collected sample count so each drain takes exactly one optimizer step (suppresses intra-drain drift; does not replace `--max-weight-staleness`) |

The reference worker caps its output queue at 1000 groups, so if training is slower
than rollout the producer eventually blocks rather than growing the queue without
bound. If the queue stays at zero, rollout is the bottleneck — scale rollout capacity
or lower per-sample generation cost.

## Off-policyness and steps per drain

Fully-async keeps a **one-optimizer-step-per-drain** invariant by default: each drained
batch is consumed in a single optimizer step. That invariant is what keeps the
intra-drain gap from compounding on top of the inter-batch weight-version lag that
`--max-weight-staleness` bounds.

Setting `--num-steps-per-rollout N` (or a direct `--global-batch-size` that implies
`N > 1` via `rollout_batch_size * n_samples_per_prompt // global_batch_size`) breaks
the invariant. The drained batch is then split into `N` optimizer steps against
log-probs snapshotted once at drain start, so step `k` trains a policy that has already
moved `k-1` steps past that snapshot.

These two gaps are **not** the same dimension and must not be added together:

| Gap | Bound | Correction |
|---|---|---|
| Rollout-vs-trainer weight-version lag | `--max-weight-staleness` | TIS (`tis` / `tis_clipfrac`) |
| Intra-drain policy drift (multi-step) | keep `num_steps_per_rollout == 1` | PPO importance ratio (`ppo_kl` / clip-frac) |

The product of the two clipped ratios is:

```text
exp(current - rollout) = importance_ratio × tis
```

where `importance_ratio` grows with the step index inside a drain and `tis` covers the
gap `--max-weight-staleness` bounds. Watch both metric pairs when multi-step is on.

`--use-dynamic-global-batch-size` resizes the global batch to the collected sample
count (rounded down to a multiple of `dp_size`), forcing exactly one optimizer step
per drain. That suppresses the *intra-drain* gap; it is **not** a substitute for
`--max-weight-staleness`, which still bounds the *rollout-vs-trainer* lag. Both remain
in effect when set together. The flag is incompatible with
`--disable-rollout-trim-samples` (the dynamic size is only computed on the trim path).

At launch, Miles warns when the derived steps-per-drain is greater than 1 and
`--max-weight-staleness` is set, and logs an info line when dynamic global batch size
already forces one step per drain.

## What to monitor

The reference worker logs progress to stdout, not wandb. Useful lines to grep for:

```text
Global worker queue size: <N>
Staleness stats: recycled=<N>, avg_staleness=<f>, max_staleness=<N>
Warning: No progress for <N>s. Queue size: <N>, Collected: <N>/<N>
```

Treat large staleness windows as a training-quality signal, not just a performance
signal. Fast [P2P weight transfer](/advanced/p2p-weight-transfer) keeps the
rollout engines closer to the latest actor weights so fewer groups get recycled by
`--max-weight-staleness`.

## Example implementation

For a complete Qwen3 launch script and worker implementation, see the
[Fully Async Rollout example](/examples/fully-async).
