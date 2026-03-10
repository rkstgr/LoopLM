**3.1 — Small-scale training run**
Trained small (100M), wikitext-103-v1, 10000 steps
Successful see
WandB: rkstgr/looplm/hteb62kc
```
→ Final checkpoint: /p/scratch/westai0047/looplm/checkpoints/checkpoints/step_0010000.pt
============================================================
                      TRAINING SUMMARY
           (first 1000 steps vs last 1000 steps)
============================================================
Metric                         First       Last          Δ
------------------------------------------------------------
  loss                        5.6090     3.8288    -1.7802
  task_loss                   5.7219     3.9550    -1.7669
  entropy                     1.1284     1.2616    +0.1332
  avg_exit_step               2.3888     2.8413    +0.4525
------------------------------------------------------------
  Per-recurrent-step losses:
  Step          First       Last          Δ
  ------------------------------------------
  t=1          5.7248     4.1338    -1.5910
  t=2          5.7275     3.9619    -1.7656
  t=3          5.7473     3.9406    -1.8067
  t=4          5.7661     3.9463    -1.8198
============================================================
Verification:
  [✓] Total loss decreased
  [✓] Task loss decreased
  [✓] Entropy not collapsed (> 0.1)
[✓] Step t=4 loss < step t=1 loss
```


**3.3 — Recurrent depth analysis**
=== Recurrent Depth vs. Accuracy ===

| Steps |  arc_easy  |
|-------|------------|
|   1   |   0.2600   |
|   2   |   0.3000   |
|   3   |   0.3100   |
|   4   |   0.3000   |

=== Recurrent Depth vs. Accuracy ===

| Steps | hellaswag  |
|-------|------------|
|   1   |   0.2700   |
|   2   |   0.2800   |
|   3   |   0.2900   |
|   4   |   0.3100   |

**6.1 — CAPO task**

srun uv run scripts/analyze.py capo --n-individuals 20000 --train-exposures 500 --model-sizes micro --loop-counts 1 4
Capo experiment
  N individuals : 20,000
  Exposures     : 500
  Model sizes   : ['micro']
  Loop counts   : [1, 4]
  Device        : auto


[capo] size=micro (3.2M)  loop=1  N=20,000  steps≈4,069
    step   586/5865  loss=0.9350
    step  1172/5865  loss=0.9074
    step  1758/5865  loss=0.9059
    step  2344/5865  loss=0.9027
    step  2930/5865  loss=0.8978
    step  3516/5865  loss=0.8950
    step  4102/5865  loss=0.8905
    step  4688/5865  loss=0.8844
    step  5274/5865  loss=0.8826
    step  5860/5865  loss=0.8817
  → bits/param=0.006  p1=20.370  p2=32.266

[capo] size=micro (3.2M)  loop=4  N=20,000  steps≈4,069

**6.2 Mano**
```md
================================================================================
                     MANO RESULTS — Aggregated across seeds
================================================================================
  Layers  Loop  Depth     Params  max_ops    N         Accuracy           Loss
  ----------------------------------------------------------------------------
       1     4      4     0.17M       10    3   0.0597±0.0102  2.1239±0.0228
       2     2      4     0.33M       10    3   0.0633±0.0031  2.1650±0.0957
       4     1      4     0.66M       10    2   0.0645±0.0021  2.3001±0.0030
================================================================================
Expected: looped models outperform non-looped at same total depth
```
