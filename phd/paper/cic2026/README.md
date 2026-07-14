# CIC 2026 Submission — Learned Continuous Tactile Guidance

Springer LNCS-style paper, deliberately disjoint from the CoRL 2026 TaCo
submission.

## Build

```bash
cd phd/paper/cic2026
latexmk -pdf main.tex
```

`llncs.cls`, `splncs04.bst`, and `aliascnt.sty` are vendored in this folder
(fetched from CTAN), so the paper compiles standalone. Current output:
12 pages.

## How this avoids conflict with the CoRL 2026 paper

| | CoRL 2026 (TaCo) | This paper (CIC 2026) |
|---|---|---|
| Command type | Discrete gestures (14-class vocabulary) | Continuous velocity guidance |
| Model | CNN–Transformer gesture classifier | Spatial-softmax CNN–GRU velocity policy |
| Supervision | Manually recorded gesture classes | Rule-based teacher (imitation, no labels) |
| Evaluation | Manipulation/trajectory user study vs. pendant/keyboard/kinesthetic | Faithful-replay robustness benchmark vs. rule pipeline under sensor corruption |
| Core claim | Embodied–discrete command channel | Learned policies beat hand-crafted rules under sensor degradation |

No text, figures, tables, or numerical results are shared with the CoRL
manuscript. The tactile skin hardware is intentionally described in two
sentences with a placeholder citation, since hardware is TaCo's
contribution, not this paper's.

## All numbers are real

Every quantitative claim comes from experiments run in this repository:

- Dataset: 8 episodes / 18,551 frames / ~5.2 min
  (`resource/ai/data/ai_direct_finger_motion/session_fixed_speed_*`)
- Model + offline metrics: `resource/ai/models/ai_direct_finger_motion/cnn_gru_spatial_softmax_augmented_20260709/`
  (RMSE 0.0145 m/s, mode acc 92.6%, R² 0.81–0.83, 234,955 params, 1.9 ms CPU inference)
- Robustness tables/figures: `robustness_benchmark_20260709_154410/`
  (new model) and `robustness_benchmark_20260709_154003/` (legacy model,
  used for the ablation table)
- Reproduce the benchmark:
  `python3 phd/script/benchmark_ai_direct_finger_motion_robustness.py`

## Placeholders you must fill (marked red in the PDF)

1. **Authors, affiliations, emails, ORCID** (title block)
2. **Skin construction sentence + hardware citation** (Sect. 3.1)
3. **Architecture figure** (Fig. 1 is a framed placeholder box —
   redraw: frames → conv stem → spatial softmax → GRU → two heads)
4. **Ablation caveat** (Table 2): current numbers compare the deployed
   avgpool/seq-8 model vs. the new spatial-softmax/seq-16 model. For
   camera-ready, retrain a single-variable ablation:
   `python3 phd/script/train_ai_direct_finger_motion.py fixed_speed_first_trial_no_rotation fixed_speed_push_focus_v1 --encoder avgpool --no-augment`
   then run the benchmark on that checkpoint and update the table.
5. **Closed-loop user study** (Limitations) — strongest possible addition;
   the paper is written so it stands without it.
6. **Multi-user / multi-unit data** (Limitations) — delete the sentence if
   unavailable.
7. **Acknowledgements + Disclosure of Interests** (end matter).

## Checklist before submission

- [ ] Verify CIC 2026 page limit and adjust (currently 12 pages incl. refs)
- [ ] Replace placeholder authors; check anonymization rules for CIC
- [ ] Fill hardware sentence; ensure the citation does not reveal the
      under-review CoRL manuscript (cite only *published* hardware work)
- [ ] Redraw the architecture figure
- [ ] Re-run single-variable ablation (item 4)
- [ ] Update `references.bib` volume/page details where marked uncertain
