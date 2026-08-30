"""Config inheritance and cache-staleness detection.

Both exist to stop the same class of failure: a run that looks like it did what
you asked and quietly did something else.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.utils.config import load_config  # noqa: E402

build_cache = importlib.import_module("build_cache")


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


BASE = """
seed: 1337
data:
  cache_dir: artifacts/cache
  artifacts_dir: artifacts
  datasets:
    toothfairy3:
      root: ToothFairy3
task:
  labels: [implant, crown, bridge]
preprocess:
  target_spacing: 1.0
  out_shape: [128, 128, 128]
  fit_mode: isotropic_pad
train:
  lr: 3.0e-5
  batch_size: 4
"""


class TestExtends:
    def test_overrides_win_and_the_rest_is_inherited(self, tmp_path):
        write(tmp_path / "base.yaml", BASE)
        write(tmp_path / "child.yaml", """
extends: base.yaml
preprocess:
  target_spacing: 0.5
  out_shape: [256, 256, 256]
""")
        cfg = load_config(tmp_path / "child.yaml")

        assert cfg.preprocess.target_spacing == 0.5
        assert cfg.preprocess.out_shape == [256, 256, 256]
        # untouched keys survive, including ones nested beside the override
        assert cfg.preprocess.fit_mode == "isotropic_pad"
        assert cfg.train.lr == 3.0e-5
        assert cfg.task.labels == ["implant", "crown", "bridge"]

    def test_lists_replace_rather_than_concatenate(self, tmp_path):
        """out_shape [256]*3 must not become a six-element list, and a label
        subset must not silently regain the labels it dropped."""
        write(tmp_path / "base.yaml", BASE)
        write(tmp_path / "child.yaml", """
extends: base.yaml
task:
  labels: [implant]
""")
        cfg = load_config(tmp_path / "child.yaml")
        assert cfg.task.labels == ["implant"]

    def test_chain_of_three(self, tmp_path):
        write(tmp_path / "base.yaml", BASE)
        write(tmp_path / "mid.yaml", "extends: base.yaml\ntrain:\n  lr: 1.0e-4\n")
        write(tmp_path / "leaf.yaml", "extends: mid.yaml\ntrain:\n  batch_size: 1\n")
        cfg = load_config(tmp_path / "leaf.yaml")
        assert cfg.train.batch_size == 1      # from leaf
        assert cfg.train.lr == 1.0e-4         # from mid
        assert cfg.seed == 1337               # from base

    def test_circular_extends_raises(self, tmp_path):
        write(tmp_path / "a.yaml", "extends: b.yaml\nseed: 1\n")
        write(tmp_path / "b.yaml", "extends: a.yaml\nseed: 2\n")
        with pytest.raises(ValueError, match="circular"):
            load_config(tmp_path / "a.yaml")

    def test_missing_base_names_the_file(self, tmp_path):
        write(tmp_path / "child.yaml", "extends: nope.yaml\nseed: 1\n")
        with pytest.raises(FileNotFoundError, match="nope.yaml"):
            load_config(tmp_path / "child.yaml")

    def test_the_real_256_config_only_overrides_resolution_and_paths(self):
        """The shipped config must inherit the task definition, not restate it --
        a copy would drift the moment task.labels or clip_window moves."""
        base = load_config("configs/default.yaml")
        cfg = load_config("configs/preprocess_256.yaml")

        assert cfg.preprocess.out_shape == [256, 256, 256]
        assert cfg.preprocess.target_spacing == 0.5
        assert str(cfg.data.cache_dir) != str(base.data.cache_dir)
        assert str(cfg.data.artifacts_dir) != str(base.data.artifacts_dir)

        # inherited, and required to stay inherited
        assert cfg.task.labels == base.task.labels
        assert cfg.preprocess.clip_mode == base.preprocess.clip_mode
        assert cfg.preprocess.clip_window == base.preprocess.clip_window
        assert cfg.preprocess.fit_mode == base.preprocess.fit_mode
        assert cfg.train.lr == base.train.lr

        # effective batch is preserved so resolution stays the only variable
        assert cfg.train.batch_size * cfg.train.accum_steps == base.train.batch_size

        # 0.5 x 256 == 1.0 x 128: same field of view, by construction
        fov = cfg.preprocess.target_spacing * cfg.preprocess.out_shape[0]
        assert fov == base.preprocess.target_spacing * base.preprocess.out_shape[0]


SETTINGS = {"out_shape": [128, 128, 128], "target_spacing": 1.0, "fit_mode": "isotropic_pad"}


class TestStaleCache:
    def test_empty_dir_is_not_stale(self, tmp_path):
        assert build_cache.stale_settings(tmp_path, SETTINGS) is None

    def test_matching_stamp_is_not_stale(self, tmp_path):
        (tmp_path / "a.npy").write_bytes(b"x")
        build_cache.write_settings(tmp_path, SETTINGS)
        assert build_cache.stale_settings(tmp_path, SETTINGS) is None

    def test_tuples_and_lists_compare_equal(self, tmp_path):
        """kwargs carry tuples; JSON stores lists. A round-trip must not look
        like a settings change or every build would rebuild everything."""
        (tmp_path / "a.npy").write_bytes(b"x")
        build_cache.write_settings(tmp_path, {"out_shape": (128, 128, 128), "clip_window": (-1000.0, 6000.0)})
        assert build_cache.stale_settings(
            tmp_path, {"out_shape": (128, 128, 128), "clip_window": (-1000.0, 6000.0)}
        ) is None

    def test_changed_resolution_is_reported_with_old_and_new(self, tmp_path):
        (tmp_path / "a.npy").write_bytes(b"x")
        build_cache.write_settings(tmp_path, SETTINGS)
        changed = build_cache.stale_settings(tmp_path, {**SETTINGS, "target_spacing": 0.5})
        assert changed == {"target_spacing": (1.0, 0.5)}

    def test_unstamped_cache_is_treated_as_unknown(self, tmp_path):
        """A cache with volumes but no stamp cannot be certified, so it rebuilds
        rather than being assumed current."""
        (tmp_path / "a.npy").write_bytes(b"x")
        assert build_cache.stale_settings(tmp_path, SETTINGS) is not None

    def test_the_shipped_1mm_cache_is_stale_against_the_256_config(self):
        """The exact accident this guards: pointing the 256 build at the 1 mm
        cache must be detected, not silently skipped as already-cached."""
        from src.utils.config import cache_dir

        base, cfg256 = load_config("configs/default.yaml"), load_config("configs/preprocess_256.yaml")
        one_mm = cache_dir(base, "toothfairy3")
        if not any(one_mm.glob("*.npy")):
            pytest.skip("1 mm cache not built on this machine")

        kwargs = {"out_shape": tuple(cfg256.preprocess.out_shape),
                  "target_spacing": cfg256.preprocess.target_spacing}
        changed = build_cache.stale_settings(one_mm, kwargs)
        assert changed is not None
        assert "target_spacing" in changed and "out_shape" in changed

    def test_stamp_is_json_and_sorted(self, tmp_path):
        build_cache.write_settings(tmp_path, {"b": 2, "a": (1, 2)})
        text = (tmp_path / build_cache.SETTINGS).read_text(encoding="utf-8")
        assert list(json.loads(text)) == ["a", "b"]
        assert json.loads(text)["a"] == [1, 2]


class TestArtifactIsolation:
    """Every shipped config must keep its outputs to itself.

    The bug this guards: configs/sites.yaml overrode data.artifacts_dir but
    inherited train.out_dir from default.yaml, so site checkpoints were written
    into the superseded detection task's run directory. A later --checkpoint
    would then silently load a model trained on a different task with a
    different number of outputs.
    """

    CONFIGS = ("configs/default.yaml", "configs/sites.yaml", "configs/preprocess_256.yaml")

    @pytest.mark.parametrize("path", CONFIGS)
    def test_run_dir_sits_under_the_artifacts_dir(self, path):
        from pathlib import Path

        cfg = load_config(path)
        art = Path(str(cfg.data.artifacts_dir)).as_posix().rstrip("/")
        out = Path(str(cfg.train.out_dir)).as_posix()
        assert out.startswith(art + "/"), (
            f"{path}: train.out_dir {out!r} is outside data.artifacts_dir {art!r} -- "
            "checkpoints would land in another task's directory"
        )

    @pytest.mark.parametrize("path", CONFIGS)
    def test_cache_dir_sits_under_the_artifacts_dir(self, path):
        from pathlib import Path

        cfg = load_config(path)
        art = Path(str(cfg.data.artifacts_dir)).as_posix().rstrip("/")
        cache = Path(str(cfg.data.cache_dir)).as_posix()
        assert cache.startswith(art + "/"), f"{path}: cache_dir escapes artifacts_dir"

    def test_the_three_configs_do_not_share_an_artifacts_dir(self):
        """Two tasks writing to one directory means one silently overwrites the
        other's labels, splits and checkpoints."""
        dirs = [str(load_config(p).data.artifacts_dir) for p in self.CONFIGS]
        assert len(set(dirs)) == len(dirs), f"configs share an artifacts dir: {dirs}"


class TestRunNaming:
    """A cross-validation round must not overwrite the previous one.

    train.py used to name every run after the model, so five folds wrote to one
    directory unless --out was passed by hand. pool_cv.py looks for cv_fold{k}
    and would then either fail outright or pool a single model against itself.

    This class used to carry its OWN copy of the naming rule and assert against
    that, so the two could drift apart while every assertion still passed. It
    now calls `train.run_name`, which is the function `main` uses.
    """

    @staticmethod
    def run_name(model_name: str, fold, synthetic: bool) -> str:
        train = importlib.import_module("train")
        return train.run_name(model_name, fold, synthetic)

    def test_each_fold_gets_its_own_directory(self):
        names = {self.run_name("vit3d", k, False) for k in range(5)}
        assert names == {f"cv_fold{k}" for k in range(5)}
        assert len(names) == 5

    def test_a_single_split_run_keeps_the_model_name(self):
        assert self.run_name("vit3d", None, False) == "vit3d"

    def test_the_synthetic_gate_never_claims_a_fold_directory(self):
        """The gate trains on planted blobs; its checkpoint must never be
        mistaken for a cross-validation round by pool_cv."""
        assert self.run_name("vit3d", 3, True) == "vit3d_synthetic"

    def test_the_name_matches_what_pool_cv_looks_for(self):
        import re
        from pathlib import Path

        source = Path("scripts/pool_cv.py").read_text(encoding="utf-8")
        pattern = re.search(r'f"cv_fold\{(\w+)\}"', source)
        assert pattern, "pool_cv.py no longer builds a cv_fold{k} directory name"
        assert self.run_name("vit3d", 2, False) == "cv_fold2"
