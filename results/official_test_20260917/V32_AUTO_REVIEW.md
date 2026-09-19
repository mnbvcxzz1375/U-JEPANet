# V3.2 P0/P1 fixes — tests ALL_PASS, launch authorized after review

## Fixes (user review of fe008f6)

| ID | Fix |
|---|---|
| **P0** B0 val crash | `train_v32.py`: A0DA uses `evaluate_word_whole_volume` (plain UNet3D) |
| **P0** batch RMS | `rms_norm_delta` **per-sample** RMS_b |
| **P1** coarse align | coarse map also **GridSample @ p_i** |
| **P1** H0≡R1 | `innov_merge(..., bias=False)`; I=0 ⇒ Δ=0 |

## Tests (40901) ALL_V32_TESTS_PASS

- B0 plain UNet val callable
- per-sample RMS independence
- coarse GridSample token count
- innov_merge bias=False + H0 S_pre/S_fuse=0
- H0 ≡ R1 forward (atol 1e-4)
- + prior init match / H0 I=0 / H1 S_pre>0 / S_fuse≈α

## Launch

B0/H0/H1 × s42/s43 after user OK; gate H1−H0 val ≳+0.003 + real-case shuffle.
