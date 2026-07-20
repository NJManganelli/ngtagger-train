"""MDMM (Route-B keras.ops port) tests: synthetic constrained problems with
known KKT points, semantics cross-checks against the pip torch `mdmm`
reference, constraint-target proxies, and the trainer/EBOPs integration.

Sign conventions follow pip mdmm: for MaxConstraint the infeasibility is
inf = max - fn - slack^2, so a binding constraint converges to a NEGATIVE
lambda with |lambda| = the KKT multiplier (min (x-2)^2 s.t. x<=1 has
KKT multiplier 2 -> lambda -> -2)."""

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402
from keras import ops  # noqa: E402

from ngtagger.train.mdmm import (  # noqa: E402
    ConstraintContext,
    EBOPsConstraint,
    EqConstraint,
    MaxConstraint,
    MDMMModel,
    build_constraints,
    make_pt_bias_fn,
    make_soft_efficiency_fn,
)


class _Scalar(keras.Model):
    """One trainable scalar x; y_pred == x for every row."""

    def __init__(self, x0=0.0):
        super().__init__(name="scalar")
        self.xvar = self.add_weight(
            name="x", shape=(), initializer=keras.initializers.Constant(x0), trainable=True
        )

    def call(self, inputs):
        return ops.reshape(self.xvar, (1, 1)) * ops.ones_like(inputs)


def _fit_scalar(constraint, steps=800, lr=0.05, optimizer=None, batch=64):
    """min mean((pred-2)^2) with one MDMM constraint on fn(ctx)=x.
    All rows identical -> every optimizer step sees the same full batch."""
    base = _Scalar(0.0)
    m = MDMMModel(base, [constraint], ["out"])
    m.compile(optimizer=optimizer or keras.optimizers.Adam(lr), loss={"out": "mse"})
    steps_per_epoch = 50
    X = np.ones((batch * steps_per_epoch, 1), dtype="float32")
    Y = np.full((batch * steps_per_epoch, 1), 2.0, dtype="float32")
    m.fit(X, {"out": Y}, batch_size=batch, epochs=steps // steps_per_epoch, verbose=0)
    return float(ops.convert_to_numpy(base.xvar)), float(ops.convert_to_numpy(constraint.lmbda))


def _x_of(ctx):
    return ops.mean(ctx.y_pred["out"])


def test_max_constraint_toy_kkt():
    # min (x-2)^2 s.t. x <= 1  ->  x* = 1, KKT multiplier 2 (lambda -> -2)
    x, lam = _fit_scalar(MaxConstraint(_x_of, 1.0, slack_init=1.0, damping=10.0), steps=400)
    assert abs(x - 1.0) < 0.02, x
    assert abs(lam + 2.0) < 0.1, lam


def test_eq_constraint_toy_kkt():
    # min (x-2)^2 s.t. x = 0.5  ->  x* = 0.5, lambda -> 2*(x*-2) = -3
    x, lam = _fit_scalar(EqConstraint(_x_of, 0.5, damping=1.0), steps=800)
    assert abs(x - 0.5) < 0.02, x
    assert abs(lam + 3.0) < 0.1, lam


def test_unconstrained_reference():
    # sanity: without the constraint the same setup reaches x = 2
    base = _Scalar(0.0)
    m = MDMMModel(base, [], ["out"])
    m.compile(optimizer=keras.optimizers.Adam(0.05), loss={"out": "mse"})
    X = np.ones((3200, 1), dtype="float32")
    m.fit(X, {"out": np.full((3200, 1), 2.0, dtype="float32")}, batch_size=64, epochs=8, verbose=0)
    assert abs(float(ops.convert_to_numpy(base.xvar)) - 2.0) < 0.05


def test_value_parity_vs_pip_mdmm():
    """Augmented-penalty value + infeasibility match pip mdmm exactly at a
    fixed (x, lambda, slack) state."""
    torch = pytest.importorskip("torch")
    pip_mdmm = pytest.importorskip("mdmm")

    x0, lam0, slack0, damping, maxv = 0.3, -0.7, 0.9, 3.0, 1.0
    tx = torch.tensor(x0)
    tcon = pip_mdmm.MaxConstraint(lambda: tx, maxv, damping=damping)
    with torch.no_grad():
        tcon.lmbda.copy_(torch.tensor(lam0))
        tcon.slack.copy_(torch.tensor(slack0))
    tret = tcon()

    base = _Scalar(x0)
    kcon = MaxConstraint(_x_of, maxv, slack_init=slack0, damping=damping)
    km = MDMMModel(base, [kcon], ["out"])
    km(np.ones((1, 1), dtype="float32"))  # build variables
    kcon.lmbda.assign(lam0)
    ctx = ConstraintContext(y_pred={"out": ops.convert_to_tensor(np.full((4, 1), x0, "float32"))})
    pen = float(ops.convert_to_numpy(kcon.penalty(ctx)))

    assert np.isclose(pen, float(tret.value.detach()), rtol=1e-6)
    assert np.isclose(float(ops.convert_to_numpy(kcon._last_inf)), float(tret.inf.detach()), rtol=1e-6)


def test_sgd_lockstep_vs_pip_mdmm():
    """With plain SGD, identical initial state and identical batches, our
    stop-gradient ascent reproduces pip mdmm's negative-lr scheme step by
    step (final x/lambda/slack agree to float precision)."""
    torch = pytest.importorskip("torch")
    pip_mdmm = pytest.importorskip("mdmm")

    lr, steps = 0.02, 400
    tx = torch.nn.Parameter(torch.tensor(0.0))
    tcon = pip_mdmm.MaxConstraint(lambda: tx, 1.0, damping=1.0)
    topt = pip_mdmm.MDMM([tcon]).make_optimizer([tx], optimizer=torch.optim.SGD, lr=lr)
    for _ in range(steps):
        topt.zero_grad()
        ret = pip_mdmm.MDMM([tcon])((tx - 2.0) ** 2)
        ret.value.backward()
        topt.step()

    base = _Scalar(0.0)
    # pip mdmm lazily initializes slack = sqrt(relu(max - fn())) = 1 at x0=0
    kcon = MaxConstraint(_x_of, 1.0, slack_init=1.0, damping=1.0)
    km = MDMMModel(base, [kcon], ["out"])
    km.compile(optimizer=keras.optimizers.SGD(lr), loss={"out": "mse"})
    steps_per_epoch = 50
    X = np.ones((8 * steps_per_epoch, 1), dtype="float32")
    Y = np.full((8 * steps_per_epoch, 1), 2.0, dtype="float32")
    km.fit(X, {"out": Y}, batch_size=8, epochs=steps // steps_per_epoch, shuffle=False, verbose=0)

    assert abs(float(ops.convert_to_numpy(base.xvar)) - float(tx.detach())) < 1e-4
    assert abs(float(ops.convert_to_numpy(kcon.lmbda)) - float(tcon.lmbda.detach())) < 1e-4
    assert abs(float(ops.convert_to_numpy(kcon.slack)) - float(tcon.slack.detach())) < 1e-4


def test_two_head_bound():
    """Toy two-head net: head B has loss_weight 0 and is driven ONLY by an
    MDMM ceiling on its mse, while head A is minimized as the main
    objective."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2048, 4)).astype("float32")
    wa, wb = rng.normal(size=4), rng.normal(size=4)
    ya = (X @ wa).astype("float32")
    yb = (X @ wb).astype("float32")

    inp = keras.Input((4,))
    h = keras.layers.Dense(16, activation="relu")(inp)
    a = keras.layers.Dense(1, name="head_a")(h)
    b = keras.layers.Dense(1, name="head_b")(h)
    base = keras.Model(inp, [a, b])

    from ngtagger.train.mdmm import make_head_loss_fn

    bound = 0.3
    con = MaxConstraint(make_head_loss_fn("head_b", loss="mse"), bound, damping=5.0, name="b_mse")
    m = MDMMModel(base, [con], ["head_a", "head_b"])
    m.compile(optimizer=keras.optimizers.Adam(0.01),
              loss={"head_a": "mse", "head_b": "mse"},
              loss_weights={"head_a": 1.0, "head_b": 0.0})
    m.fit(X, {"head_a": ya, "head_b": yb}, batch_size=256, epochs=40, verbose=0)

    pa, pb = base.predict(X, verbose=0)
    mse_a = float(np.mean((pa[:, 0] - ya) ** 2))
    mse_b = float(np.mean((pb[:, 0] - yb) ** 2))
    assert mse_b <= bound + 0.1, mse_b  # ceiling held despite zero loss weight
    assert mse_a < 0.5 * float(np.var(ya)), mse_a  # main objective still minimized


def test_pt_bias_fn_value():
    fn = make_pt_bias_fn("pT_output")
    pred = np.array([[1.1], [0.9], [1.2], [1.0]], dtype="float32")
    true = np.array([1.0, 1.0, 1.0, 1.0], dtype="float32")
    ctx = ConstraintContext(y_pred={"pT_output": ops.convert_to_tensor(pred)},
                            y_true={"pT_output": ops.convert_to_tensor(true)})
    got = float(ops.convert_to_numpy(fn(ctx)))
    assert np.isclose(got, abs(np.mean(pred[:, 0] / true - 1.0)), rtol=1e-6)


def test_soft_efficiency_fn_value():
    labels = ["sig", "bkg1", "bkg2"]
    fn = make_soft_efficiency_fn("jet_id_output", select_classes=["bkg1", "bkg2"],
                                 score_classes=["sig"], class_labels=labels,
                                 threshold=0.5, temperature=0.05)
    logits = np.array([[5.0, 0.0, 0.0],   # sig-like, true bkg1 -> passes WP
                       [-5.0, 5.0, 0.0],  # bkg-like, true bkg2 -> fails WP
                       [5.0, 0.0, 0.0]],  # sig-like, true sig  -> not selected
                      dtype="float32")
    y = np.eye(3)[[1, 2, 0]].astype("float32")
    ctx = ConstraintContext(y_pred={"jet_id_output": ops.convert_to_tensor(logits)},
                            y_true={"jet_id_output": ops.convert_to_tensor(y)})
    got = float(ops.convert_to_numpy(fn(ctx)))
    assert 0.45 < got < 0.55, got  # one of the two background jets passes


def test_build_constraints_validation():
    with pytest.raises(ValueError, match="items"):
        build_constraints({"items": []}, ["out"])
    with pytest.raises(ValueError, match="unknown constraint target"):
        build_constraints({"items": [{"target": "nope", "value": 1}]}, ["out"])
    with pytest.raises(ValueError, match="unknown constraint type"):
        build_constraints({"items": [{"target": "pt_bias", "type": "huh", "value": 1}]}, ["out"])


def test_ebops_constraint_unit():
    """Dual update mechanics with a plugged ebops_fn (the adapter point)."""
    cb = EBOPsConstraint(budget=100.0, lambda_lr=0.1, damping=2.0, ebops_fn=lambda m: 300.0)
    cb.set_model(keras.Sequential([keras.layers.Dense(1)]))
    cb.on_train_batch_end(0)
    g = 300.0 / 100.0 - 1.0
    assert np.isclose(cb.lmbda, 0.1 * g)
    assert np.isclose(cb.last_inf, g)
    # under budget: lambda decays and clamps at 0
    cb.ebops_fn = lambda m: 10.0
    for i in range(20):
        cb.on_train_batch_end(i + 1)
    assert cb.lmbda == 0.0


def _tiny_hgq2(tmp_path, extra_cfg):
    import yaml

    from ngtagger.models.base import ModelRegistry

    cfg = {
        "model": "DeepSetHGQ2",
        "run_config": {"verbose": 0},
        "model_config": {"name": "tiny", "conv1d_layers": [4], "conv1d_parallelisation_factor": [16],
                         "classification_layers": [8], "regression_layers": [4], "beta": 1e-8},
        "training_config": {"weight_method": "onlyclass", "validation_split": 0.2, "epochs": 2,
                            "batch_size": 128, "learning_rate": 0.01, "loss_weights": [1.0, 1.0]},
        "firmware_config": {"project_name": "tiny_test"},
    }
    for k, v in extra_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    model = ModelRegistry.create("DeepSetHGQ2", str(tmp_path / "out"))
    model.load_yaml(str(p))
    return model, str(p)


def _synthetic(n=384, n_feat=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 16, n_feat)).astype("float32")
    y = np.eye(8)[rng.integers(0, 8, n)]
    pt = rng.uniform(0.3, 2.0, n).astype("float32")
    return X, y, pt


def test_ebops_constraint_engages_on_hgq2(tmp_path):
    """Impossible budget -> lambda grows and every QLayer beta is re-assigned
    to scale*(lambda + damping*g)."""
    pytest.importorskip("hgq")
    model, _ = _tiny_hgq2(tmp_path, {"constraints": {"items": [
        {"target": "ebops", "value": 10.0, "lambda_lr": 1e-3, "damping": 0.0},
    ]}})
    X, y, pt = _synthetic()
    model.class_labels = [f"c{i}" for i in range(8)]
    model.feature_names = [f"f{i}" for i in range(20)]
    model.build((16, 20), 8)
    model.compile()
    model.fit(X, y, pt, validation_split=0.2, seed=1)

    cb = model._mdmm_callbacks[0]
    assert isinstance(cb, EBOPsConstraint)
    assert cb.lmbda > 0.0
    assert cb.last_ebops > 10.0
    betas = [float(ops.convert_to_numpy(la._beta)) for la in model.model._flatten_layers()
             if getattr(la, "_beta", None) is not None]
    assert betas and all(np.isclose(b, cb.lmbda, rtol=1e-5) for b in betas)
    # metrics surfaced for mlflow
    cm = model.constraint_metrics()
    assert cm["lambda_ebops"] == cb.lmbda and cm["ebops_total"] == cb.last_ebops


def test_run_training_with_constraints(tmp_path, monkeypatch):
    """Full trainer path (dataset override, no nano files): constraints
    section + charge head, mlflow redirected to tmp."""
    pytest.importorskip("hgq")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")

    from ngtagger.train.trainer import run_training

    _, cfg_path = _tiny_hgq2(tmp_path, {
        "experiment": "ngtagger-test",
        "model_config": {"charge_layers": [4]},
        "training_config": {"loss_weights": [1.0, 1.0, 0.0]},
        "constraints": {"damping": 1.0, "items": [
            {"target": "pt_bias", "type": "max", "value": 0.01},
            {"target": "head_loss", "head": "charge_output", "type": "max", "value": 0.9},
        ]},
    })
    X, y, pt = _synthetic()
    n = len(X)
    qcls = np.random.default_rng(1).integers(0, 3, n)
    ds = {
        "X_train": X[: n - 64], "y_train": y[: n - 64], "pt_train": pt[: n - 64],
        "truth_pt_train": pt[: n - 64], "reco_pt_train": np.full(n - 64, 50.0),
        "charge_train": np.eye(3)[qcls[: n - 64]],
        "X_test": X[n - 64:], "y_test": y[n - 64:], "pt_test": pt[n - 64:],
        "truth_pt_test": pt[n - 64:], "reco_pt_test": np.full(64, 50.0),
        "charge_test": np.eye(3)[qcls[n - 64:]],
        "feature_names": [f"f{i}" for i in range(20)],
        "class_labels": [f"c{i}" for i in range(8)],
        "charge_class_labels": ["qminus", "neutral", "qplus"],
    }
    model = run_training(cfg_path, files=[], output_dir=str(tmp_path / "run_out"), dataset=ds)
    assert (tmp_path / "run_out" / "model.keras").exists()
    cm = model.constraint_metrics()
    assert "lambda_pt_bias" in cm and "lambda_head_loss" in cm
    assert "lambda_pt_bias" in model.history.history
