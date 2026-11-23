"""Replay a numerical bound from the Supplement using the saved inputs."""
import argparse
import hashlib
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path, PurePath


def main():
    root = Path(__file__).resolve().parent
    old = root / "research/stable-light-rings"
    new = root / "research/self-radiation"
    stages = {
        "center": (old, "regular_center", []),
        "exterior": (old, "asymptotic_exterior", []),
        "inverse": (old, "finite_static_inverse",
                    ["--reference", str(old / "data/s08a06/finite-static-reference.npz")]),
        "defect": (old, "background_defect", []),
        "star": (old, "global_background", []),
        "optics": (old, "relative_optics", []),
        "frequencies": (old, "orbital_frequency_monotonicity", ["--digits", "50"]),
        "polar": (old, "polar_transverse", ["--digits", "50"]),
        "resonance": (new, "slow_orbit_reduction", ["--digits", "50", "--interval-only"]),
        "tide": (new, "tidal_field_error", ["--digits", "50"]),
        "observation": (new, "iterated_tidal_observation", ["--digits", "50", "--interval-only"]),
        "orbit": (new, "azimuth_orbit", ["--digits", "50"]),
        "overlap": (new, "resonant_tidal_overlap", ["--digits", "50", "--refine", "2"]),
        "phase": (new, "stationary_phase", ["--digits", "50"]),
        "core": (new, "frequency_regular_polar", ["--digits", "50"]),
        "threshold": (new, "low_frequency_quadrupole", []),
        "outgoing": (new, "coupled_outgoing_space", ["--digits", "50"]),
        "forcing": (new, "forced_polar_continuation", ["--digits", "50"]),
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=stages, default="overlap",
                        help="Recheck one stage using the archived inputs (default: overlap).")
    stage = parser.parse_args().stage
    if not __debug__:
        parser.error("Use Python with assertions enabled.")
    for name, digest in json.loads((root / "inventory.json").read_text()).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed archive member: {name}")
    if os.name != "nt":
        # Dependency maps in the archived certificates use Windows paths.
        join = PurePath.__truediv__
        def record_path(self, part):
            if isinstance(part, str) and part.startswith("research\\"):
                part = part.replace("\\", "/")
            return join(self, part)
        PurePath.__truediv__ = record_path
    directory, name, options = stages[stage]
    source = directory / f"check_{name}.py"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source.parent))
    if stage == "threshold":
        # Use the calculation directly; paper source archives are not inputs.
        from mpmath import iv
        from check_low_frequency_quadrupole import algebra_checks, conditional_power, exterior_tail
        iv.dps = 50
        print(json.dumps(dict(checks=algebra_checks(), exterior=exterior_tail(),
                              conditional_prediction=conditional_power()), indent=2))
        return
    with tempfile.TemporaryDirectory() as folder:
        result = Path(folder) / f"{stage}.json"
        sys.argv = [str(source), *options, "--output", str(result)]
        runpy.run_path(str(source), run_name="__main__")
        data = json.loads(result.read_text())
        if stage == "orbit" and data["status"] != "azimuth_orbit_ball_verified":
            raise ValueError("The resonant half-orbit enclosure did not close.")
        if stage != "overlap":
            return
        lo, hi = data["complete_overlap"]
        if not (data["cells"] == 512 and 2.029e-7 < lo <= hi < 4.118e-7):
            raise ValueError(f"Published enclosure not reproduced: {lo}, {hi}")
        print(f"512-cell static enclosure: [{lo:.12e}, {hi:.12e}]")


if __name__ == "__main__":
    main()
