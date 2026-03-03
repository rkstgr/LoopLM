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

uv run scripts/analyze.py capo --n-individuals 200 --train-exposures 20 --model-sizes micro --loop-counts 1 4 --batch-size 8 --seq-len 64
Capo experiment
  N individuals : 200
  Exposures     : 20
  Model sizes   : ['micro']
  Loop counts   : [1, 4]
  Device        : auto


[capo] size=micro (3.2M)  loop=1  N=200  steps≈312
    step    44/449  loss=8.2363
    step    88/449  loss=3.6798
    step   132/449  loss=2.7903
    step   176/449  loss=1.8244
    step   220/449  loss=1.3435
    step   264/449  loss=1.1793
    step   308/449  loss=1.0860
    step   352/449  loss=1.0683
    step   396/449  loss=1.0211
    step   440/449  loss=1.0313
  → bits/param=0.000  p1=20.185  p2=33.628

[capo] size=micro (3.2M)  loop=4  N=200  steps≈312
    step    44/449  loss=8.2052
    step    88/449  loss=3.6916
    step   132/449  loss=3.1650
    step   176/449  loss=2.3041
    step   220/449  loss=1.6212
    step   264/449  loss=1.2890
    step   308/449  loss=1.1576
    step   352/449  loss=1.1125
    step   396/449  loss=1.0594
    step   440/449  loss=1.0482
  → bits/param=0.000  p1=20.648  p2=34.545

=================================================================
                CAPO RESULTS — Knowledge Capacity                
=================================================================
  Size       Params  Loop        N   bits/param       p1       p2
  -------------------------------------------------------------
  micro       3.2M     1      200        0.000   20.185   33.628
  micro       3.2M     4      200        0.000   20.648   34.545
=================================================================
Expected: bits/param ≈ 2.0 for both loop=1 and loop=4

Results saved to runs/capo/capo_results.csv
