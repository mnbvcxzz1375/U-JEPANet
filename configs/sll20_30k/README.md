# WORD-SLL 20% · 30k supervised updates · seed 42

Protocol (locked):

- 100 train = 20 labeled + 80 unlabeled (frozen `seed_split=20260916`)
- 20 official `imagesVal` whole-volume sliding-window eval
- `imagesTs` sealed
- Dual loader: every step `L_seg(B_L)` from 20L; A2/A3 also `L_JEPA(B_J)` from 100 train
- Equal **labeled exposure**: 30k seg updates for all arms
- JEPA schedule: 0–3000 λ=0, 3000–6000 ramp to 0.3, then hold
- Model selection: validation mean foreground Dice only

See repo `EXPERIMENT_PROTOCOL.md` and `results/` after runs.
