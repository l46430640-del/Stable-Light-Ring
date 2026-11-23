# How Trapped Gravitational Waves Radiate near a Stable Light Ring

Calculation code and saved results for the Letter and its Supplement.
The main numerical result is the positive static tidal pairing of the
resonant orbit: `2.029e-7 < C_25 < 4.118e-7`. The finite-time radiation
and equal-energy comparison follow analytically in the Supplement.

## Reproduce the static pairing

From the repository root, with Python 3.11 or 3.12:

    python -m pip install -r requirements.txt
    python replay.py

The default stage evaluates the final interval integral using the saved
stellar, orbital and tidal bounds. Their input arrays are in NPZ format,
and the bounds and error budgets are in JSON. The final record is
[resonant-tidal-overlap-50.json](research/self-radiation/data/resonant-tidal-overlap-50.json).
Each stage checks the file inventory and uses temporary space for its output.

| Supplement | Calculation | Stages, in dependency order |
| --- | --- | --- |
| S1 | Equilibrium and error bounds for the S08A06 star | `center`, `exterior`, `inverse`, `defect`, `star` |
| S2 | Trapping, resonant orbit and its static tidal pairing | `optics`, `frequencies`, `polar`, `resonance`, `tide`, `observation`, `orbit`, `overlap` |
| S3 | Polar field in the core, low-frequency matching and periodic forcing | `phase`, `core`, `threshold`, `outgoing`, `forcing` |

To repeat one stage, use e.g. `python replay.py --stage core`. Each stage
reads the distributed records, so it can be run on its own.
The signed interval in S2 is a static response; the periodic power in S3
is for a prescribed source. Neither record is a simulated waveform.

## Files

- [Stellar calculations](research/stable-light-rings/): equilibrium solver,
  radial operators, interval bounds and reference profiles.
- [Wave calculations](research/self-radiation/): orbit, tidal response,
  outgoing quadrupole, and algebra used in S1-S5. The algebraic outputs are
  collected in [proof-identities.json](research/self-radiation/data/proof-identities.json)
  under the name of the corresponding calculation. The S4 variation
  identities are in [check_s4_finite_variations.py](research/self-radiation/check_s4_finite_variations.py).
- [Figures](figures/): the two Letter figures in SVG and PDF.
  `python figures/draw.py` exports them with Inkscape on the executable path.

The original numerical certificates retain their SHA-256 input references.
Two references in the preliminary channel record identify the public TeX
source of [arXiv:1703.09069](https://arxiv.org/abs/1703.09069); the paper
source is not needed by the replay stages.
