"""EzklCircuit: build, prove and verify one EZKL (Halo2 + KZG) circuit from a PyTorch module.

A circuit lives in its own directory:
  model.onnx  settings.json  model.compiled  pk.key (proving key, large)  vk.key (verification key)
Verifying needs only settings.json, vk.key and the public KZG reference string (downloaded once by
ezkl.get_srs from the public trusted setup), so those are what gets published.

Public values are always read from the proof's `instances` - the field elements the proof binds -
never from its `pretty_public_inputs`, which are display copies anyone could edit.
"""

import asyncio
import hashlib
import inspect
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()  # EZKL proving is memory-heavy; one proof at a time per process


class ZkError(RuntimeError):
    """EZKL failed (it reports some failures as Rust panics, which are not Exceptions)."""


def _ensure_env() -> None:
    """EZKL keeps the downloaded KZG reference strings under $EZKL_REPO_PATH (default $HOME/.ezkl)
    and panics if HOME is unset - which it is for services started on Windows."""
    home = str(Path.home())
    os.environ.setdefault("HOME", home)
    os.environ.setdefault("EZKL_REPO_PATH", str(Path(home) / ".ezkl"))


class ZkUnavailable(RuntimeError):
    """EZKL is not installed or the circuit has not been set up (scripts/zk_setup.py)."""


def _ezkl():
    try:
        import ezkl
    except ImportError as e:  # optional dependency: backend/requirements-zk.txt
        raise ZkUnavailable("EZKL is not installed: pip install -r backend/requirements-zk.txt") from e
    return ezkl


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call an EZKL function inside an event loop: some (get_srs) are coroutines that must be
    created in a running loop, others are plain. Uses a helper thread if this thread already
    runs a loop."""

    _ensure_env()

    async def go() -> Any:
        try:
            out = fn(*args, **kwargs)
            return await out if inspect.isawaitable(out) else out
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # pyo3 PanicException derives from BaseException
            raise ZkError(f"EZKL {getattr(fn, '__name__', 'call')} failed: {e}") from None

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(go())
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["v"] = asyncio.run(go())
        except BaseException as e:  # re-raised in the caller's thread
            box["e"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box.get("v")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EzklCircuit:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)

    # ------------------------------------------------------------------ paths
    @property
    def onnx(self) -> Path:
        return self.dir / "model.onnx"

    @property
    def settings_path(self) -> Path:
        return self.dir / "settings.json"

    @property
    def compiled(self) -> Path:
        return self.dir / "model.compiled"

    @property
    def pk(self) -> Path:
        return self.dir / "pk.key"

    @property
    def vk(self) -> Path:
        return self.dir / "vk.key"

    def ready_to_prove(self) -> bool:
        return all(p.exists() for p in (self.settings_path, self.compiled, self.pk))

    def ready_to_verify(self) -> bool:
        return self.settings_path.exists() and self.vk.exists()

    @property
    def settings(self) -> dict[str, Any]:
        return json.loads(self.settings_path.read_text(encoding="utf-8"))

    def describe(self) -> dict[str, Any]:
        s = self.settings
        return {"logrows": s["run_args"]["logrows"], "input_scale": s["run_args"]["input_scale"],
                "param_scale": s["run_args"]["param_scale"], "input_visibility": s["run_args"]["input_visibility"],
                "vk_sha256": file_sha256(self.vk), "settings_sha256": file_sha256(self.settings_path)}

    # ------------------------------------------------------------------ build
    def build(self, module, sample: list[list[float]], input_visibility: str, scale: int,
              logrows: int | None = None, lookup_range: tuple[int, int] | None = None) -> dict[str, Any]:
        """Export to ONNX, generate settings at a fixed scale, compile, fetch the SRS, run setup.
        Returns timings. Fixed scales (no calibration) keep the build reproducible."""
        import torch

        ezkl = _ezkl()
        self.dir.mkdir(parents=True, exist_ok=True)
        x = torch.tensor(sample, dtype=torch.float32)
        torch.onnx.export(module, x, str(self.onnx), input_names=["input"], output_names=["output"],
                          opset_version=17, dynamo=False)
        args = ezkl.PyRunArgs()
        args.input_visibility = input_visibility
        args.output_visibility = "public"
        args.param_visibility = "fixed"  # weights are circuit constants: the vk commits to them
        args.input_scale = scale
        args.param_scale = scale
        if logrows:
            args.logrows = logrows
        if lookup_range:
            args.lookup_range = lookup_range
        t0 = time.perf_counter()
        _call(ezkl.gen_settings, str(self.onnx), str(self.settings_path), py_run_args=args)
        _call(ezkl.compile_circuit, str(self.onnx), str(self.compiled), str(self.settings_path))
        _call(ezkl.get_srs, str(self.settings_path))
        t1 = time.perf_counter()
        _call(ezkl.setup, str(self.compiled), str(self.vk), str(self.pk))
        return {"compile_s": round(t1 - t0, 1), "setup_s": round(time.perf_counter() - t1, 1)}

    # ------------------------------------------------------------------ prove / verify
    def prove(self, inputs: list[float]) -> dict[str, Any]:
        """Proof (EZKL JSON) for one input vector, plus timing."""
        if not self.ready_to_prove():
            raise ZkUnavailable(f"circuit in {self.dir} is not set up; run backend/scripts/zk_setup.py")
        ezkl = _ezkl()
        with _LOCK, tempfile.TemporaryDirectory(prefix="zk-") as tmp:
            tmp = Path(tmp)
            (tmp / "input.json").write_text(json.dumps({"input_data": [list(map(float, inputs))]}))
            t0 = time.perf_counter()
            _call(ezkl.gen_witness, str(tmp / "input.json"), str(self.compiled), str(tmp / "witness.json"))
            _call(ezkl.prove, str(tmp / "witness.json"), str(self.compiled), str(self.pk), str(tmp / "proof.json"))
            proof = json.loads((tmp / "proof.json").read_text())
        return {"proof": proof, "prove_seconds": round(time.perf_counter() - t0, 2)}

    def verify(self, proof: dict[str, Any]) -> bool:
        if not self.ready_to_verify():
            raise ZkUnavailable(f"no verification key in {self.dir}")
        ezkl = _ezkl()
        with tempfile.TemporaryDirectory(prefix="zkv-") as tmp:
            p = Path(tmp) / "proof.json"
            p.write_text(json.dumps(proof))
            try:
                return bool(_call(ezkl.verify, str(p), str(self.settings_path), str(self.vk)))
            except Exception:  # malformed or tampered proofs raise instead of returning False
                return False

    # ------------------------------------------------------------------ public values
    def public_values(self, proof: dict[str, Any]) -> dict[str, Any]:
        """Decode the proof's bound instances: {"input_hash": hex | None, "outputs": [float]}."""
        ezkl = _ezkl()
        s = self.settings
        flat = [v for col in proof["instances"] for v in col]
        n_out = _numel(s["model_instance_shapes"][-1])  # the last instance shape is the output's
        hashed = self.input_hashed
        expected = (1 if hashed else _numel(s["model_instance_shapes"][0])) + n_out
        if len(flat) != expected:
            raise ValueError(f"proof has {len(flat)} public values, this circuit expects {expected}")
        head, outputs = flat[:-n_out], flat[-n_out:]
        return {"input_hash": head[0] if hashed else None,
                "inputs": None if hashed else [ezkl.felt_to_float(v, s["run_args"]["input_scale"]) for v in head],
                "outputs": [ezkl.felt_to_float(v, s["model_output_scales"][0]) for v in outputs]}

    @property
    def input_hashed(self) -> bool:
        vis = self.settings["run_args"]["input_visibility"]
        return isinstance(vis, dict) and "Hashed" in vis

    def input_hash(self, inputs: list[float]) -> str:
        """Poseidon commitment to an input vector, in the same form as the proof's first instance -
        lets someone who knows the private input check it is the one that was proven."""
        ezkl = _ezkl()
        scale = self.settings["run_args"]["input_scale"]
        return _call(ezkl.poseidon_hash, [ezkl.float_to_felt(float(v), scale) for v in inputs])[0]


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n
