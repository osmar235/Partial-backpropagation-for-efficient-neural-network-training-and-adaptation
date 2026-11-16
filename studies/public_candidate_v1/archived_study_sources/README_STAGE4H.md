# AGRLS Stage 4H — cheap microprobe forward gate

## Why this stage exists
Stage 4G established two facts on **seen/development** Fashion-MNIST seeds 64–72:

1. Forward gating had real decision value: full-batch FGGR finished at a median **+0.18 pp vs Adam**, while blind periodic stale-gradient reuse finished at **−1.02 pp vs Adam**. The median FGGR-minus-periodic accuracy advantage was therefore about **+1.16 pp**.
2. The full-batch gate was too expensive: its stale-trial acceptance rate was only about **15.7%**, giving only **13.5% fewer backwards**, while calibrated NN cost increased by about **14.6%**.

For a gate that costs one additional full-batch trial forward, the expected compute break-even condition is approximately

`accept_rate > F / B`

(or equivalently, using the exact accounting in the code, accepted saved full-gradient cost must exceed all gate-probe forward costs). Stage 4G was below that regime.

## Stage 4H hypothesis
Keep the **forward-first / backward-if-needed** mechanism, but make the gate much cheaper.

On a reuse-eligible new batch, Stage 4H samples a fixed **32-example probe subset** from that same 256-example mini-batch and evaluates:

- probe loss at the current parameters;
- probe loss at the stale Adam displacement candidate.

If the candidate decreases probe loss, commit the stale displacement and skip the full-batch forward+backward. If it fails, pay for a normal full-batch F+B and take a fresh Adam update.

No loss is compared across mini-batches. Maximum one consecutive skipped backward remains enforced.

## Why probe size 32
This is a development choice derived from the Stage 4G observed acceptance rate and measured forward/backward cost ratio, not selected from Stage 4H results. A 32/256 probe gives substantial cost margin even if the stale-trial acceptance rate falls relative to Stage 4G. The code separately calibrates the actual 32-example forward cost on the user's GPU; it does **not** assume linear scaling with batch size.

## Algorithms
- `adam_tensorwise`: ordinary Adam, one F+B per update.
- `periodic_k2_noquery`: strong blind baseline. Alternate fresh Adam updates with stale-displacement reuse; the reuse update performs **no new-batch NN query**.
- `fggr_microprobe32`: Stage 4H forward gate described above.

The periodic baseline is intentionally stronger in compute accounting than Stage 4G's diagnostic periodic baseline because a truly blind reuse step does not need a forward merely for logging.

## Primary development questions
1. Does the microprobe gate preserve a meaningful accuracy advantage over blind periodic reuse?
2. Is its observed gate acceptance rate above the **measured per-seed break-even acceptance rate**?
3. Does it reduce calibrated NN compute versus Adam while retaining near-Adam accuracy?

Seeds 64–72 are already seen and therefore this stage is **development evidence only**.

## Command
```cmd
C:\Users\osmar\projects\.venv\Scripts\activate
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
run_stage4h_microprobe_gate_seen_seeds_64_to_72.cmd
```

Direct command:
```cmd
python agrls_stage4h_microprobe_gate_batch.py --dataset fashionmnist --train-subset 10000 --eval-subset 5000 --batch-size 256 --probe-size 32 --updates 400 --seeds 64 65 66 67 68 69 70 71 72 --device cuda --adam-lr 0.001 --out stage4h_microprobe_gate_seen_seeds_64_to_72
```

## Important output files
- `REPORT.txt`
- `aggregate_paired_summary.csv`
- `paired_vs_adam.csv`
- `microprobe_vs_periodic_head_to_head.csv`
- per-seed summaries and traces
