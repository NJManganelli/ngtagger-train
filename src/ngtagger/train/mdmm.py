"""Backend-agnostic keras.ops port of the Modified Differential Multiplier
Method (Platt & Barr, "Constrained Differential Optimization", 1988) --
"Route B" of docs/model-space-study.md B.2.2.

Semantics mirror the pip ``mdmm`` package (torch-only, kept installed as the
reference implementation): each constraint contributes

    penalty_i = scale_i * (lambda_i * g_i + damping_i / 2 * g_i**2)

to the loss, where g_i is the *infeasibility* (pip-mdmm sign conventions):

    eq       : g = value - fn(x)
    max      : g = max - fn(x) - slack**2      (slack trainable)
    min      : g = fn(x) - min - slack**2      (slack trainable)
    max_hard : g = min(fn(x), max) - fn(x)
    min_hard : g = max(fn(x), min) - fn(x)

Model parameters and slacks perform gradient DESCENT on the penalty while
the multipliers lambda_i perform gradient ASCENT.  The pip package realizes
the ascent by giving the lambdas a negative learning rate; here it is done
backend-agnostically with a stop-gradient split so ONE ordinary optimizer
drives every variable:

    penalty = scale * ( sg(lambda)*g + damping/2*g**2      # theta/slack part
                        + sg(lambda*g) - lambda*sg(g) )    # value-neutral,
                                                           # d/dlambda = -scale*g

The extra pair of stop-gradient terms cancels in value (the reported loss is
exactly f + sum_i scale_i*(lambda_i*g_i + damping_i/2*g_i^2), as in pip mdmm)
but flips the lambda gradient to -scale*g, i.e. plain descent implements the
ascent.  With the same optimizer/lr/initial state the update sequence is
identical to pip mdmm's negative-lr scheme (tests/test_mdmm.py checks this
numerically against the installed package).

Known deviation from pip mdmm: slack variables are initialized to a constant
(``slack_init``, default 0) at build time instead of lazily from the first
fn() value -- lazy nan-initialization does not survive graph tracing.  Same
fixed points; transiently different trajectories (the cross-check test passes
an explicit slack_init to match).
"""

from __future__ import annotations

import keras
from keras import ops


class ConstraintContext:
    """Per-batch view handed to constraint functions.

    y_true / y_pred are dicts keyed by output name when the model has named
    outputs (the MDMMModel wrapper guarantees this for multi-head models).
    """

    __slots__ = ("x", "y_true", "y_pred", "sample_weight", "model")

    def __init__(self, x=None, y_true=None, y_pred=None, sample_weight=None, model=None):
        self.x = x
        self.y_true = y_true
        self.y_pred = y_pred
        self.sample_weight = sample_weight
        self.model = model


class Constraint:
    """Base MDMM constraint: fn(ctx) -> scalar tensor, plus multiplier state."""

    def __init__(self, fn, *, scale: float = 1.0, damping: float = 1.0, name: str | None = None):
        self.fn = fn
        self.scale = float(scale)
        self.damping = float(damping)
        self.name = name or type(self).__name__.lower()
        self.lmbda = None
        self._last_inf = None

    def build(self, host: keras.Model):
        self.lmbda = host.add_weight(
            name=f"lambda_{self.name}", shape=(), initializer="zeros", trainable=True
        )
        self._last_inf = host.add_weight(
            name=f"inf_{self.name}", shape=(), initializer="zeros", trainable=False
        )

    def infeasibility(self, fn_value):
        raise NotImplementedError

    def penalty(self, ctx: ConstraintContext):
        g = ops.cast(self.infeasibility(self.fn(ctx)), self.lmbda.dtype)
        self._last_inf.assign(g)
        value_term = ops.stop_gradient(self.lmbda) * g + 0.5 * self.damping * g * g
        # value-neutral pair whose only gradient is d/dlambda = -g (ascent via descent)
        ascent_term = ops.stop_gradient(self.lmbda * g) - self.lmbda * ops.stop_gradient(g)
        return self.scale * (value_term + ascent_term)


class EqConstraint(Constraint):
    def __init__(self, fn, value, **kw):
        super().__init__(fn, **kw)
        self.value = float(value)

    def infeasibility(self, fn_value):
        return self.value - fn_value


class _SlackMixin:
    def __init__(self, fn, slack_init: float = 0.0, **kw):
        super().__init__(fn, **kw)
        self.slack_init = float(slack_init)
        self.slack = None

    def build(self, host):
        super().build(host)
        self.slack = host.add_weight(
            name=f"slack_{self.name}",
            shape=(),
            initializer=keras.initializers.Constant(self.slack_init),
            trainable=True,
        )


class MaxConstraint(_SlackMixin, Constraint):
    """fn(x) <= max, slack-variable form (pip mdmm MaxConstraint)."""

    def __init__(self, fn, max_value, slack_init: float = 0.0, **kw):
        super().__init__(fn, slack_init=slack_init, **kw)
        self.max_value = float(max_value)

    def infeasibility(self, fn_value):
        return self.max_value - fn_value - self.slack * self.slack


class MinConstraint(_SlackMixin, Constraint):
    """fn(x) >= min, slack-variable form (pip mdmm MinConstraint)."""

    def __init__(self, fn, min_value, slack_init: float = 0.0, **kw):
        super().__init__(fn, slack_init=slack_init, **kw)
        self.min_value = float(min_value)

    def infeasibility(self, fn_value):
        return fn_value - self.min_value - self.slack * self.slack


class MaxConstraintHard(Constraint):
    """fn(x) <= max without a slack variable (pip mdmm MaxConstraintHard)."""

    def __init__(self, fn, max_value, **kw):
        super().__init__(fn, **kw)
        self.max_value = float(max_value)

    def infeasibility(self, fn_value):
        return ops.minimum(fn_value, self.max_value) - fn_value


class MinConstraintHard(Constraint):
    """fn(x) >= min without a slack variable (pip mdmm MinConstraintHard)."""

    def __init__(self, fn, min_value, **kw):
        super().__init__(fn, **kw)
        self.min_value = float(min_value)

    def infeasibility(self, fn_value):
        return ops.maximum(fn_value, self.min_value) - fn_value


class MDMMModel(keras.Model):
    """Training-time wrapper: base multi-output model + MDMM constraints.

    All weights are shared with ``base``; after fit() the plain base model
    carries the constrained-trained weights, so save/export the base as usual
    (the wrapper is never serialized).  Outputs are re-packed into a dict
    keyed by ``output_names`` so dict-keyed compile losses / metrics /
    targets work unchanged on the wrapper regardless of subclassed-model
    output naming.

    Metric/lambda visibility: use MDMMLogger to inject per-epoch
    ``lambda_<name>`` / ``inf_<name>`` into the history logs (the raw values
    live in host variables and are readable at any time).
    """

    def __init__(self, base: keras.Model, constraints, output_names, **kw):
        super().__init__(name=f"{base.name}_mdmm", **kw)
        self.base = base
        self.mdmm_constraints = list(constraints)
        self.output_names_list = list(output_names)
        for c in self.mdmm_constraints:
            c.build(self)

    def call(self, x, training=None):
        out = self.base(x, training=training)
        if not isinstance(out, (list, tuple)):
            out = [out]
        return dict(zip(self.output_names_list, out))

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        loss = super().compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=training
        )
        ctx = ConstraintContext(
            x=x, y_true=y, y_pred=y_pred, sample_weight=sample_weight, model=self.base
        )
        for c in self.mdmm_constraints:
            loss = loss + c.penalty(ctx)
        return loss


class MDMMLogger(keras.callbacks.Callback):
    """Adds lambda_<name> and inf_<name> (last-batch infeasibility) to the
    epoch logs so they land in History (and any downstream logger)."""

    def __init__(self, mdmm_model: MDMMModel):
        super().__init__()
        self._mdmm = mdmm_model

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for c in self._mdmm.mdmm_constraints:
            logs[f"lambda_{c.name}"] = float(ops.convert_to_numpy(c.lmbda))
            logs[f"inf_{c.name}"] = float(ops.convert_to_numpy(c._last_inf))


class EBOPsConstraint(keras.callbacks.Callback):
    """MDMM max-constraint on the HGQ2 total EBOPs (the study's flagship
    EBOPs-as-budget constraint, B.2.2), actuated through the per-layer
    NON-trainable ``beta`` variables -- the same actuator as
    hgq.utils.sugar.BaseBetaPID, but with Platt-Barr multiplier dynamics
    instead of a PID:

        g_t      = ebops_t / budget - 1          (relative=True; else ebops - budget)
        lambda_t = max(0, lambda_{t-1} + lambda_lr * g_t)
        beta_l  := scale * (lambda_t + damping * max(g_t, 0))   for every QLayer

    No extra gradient wiring is needed: HGQ2 layers already add
    ``beta * ebops(theta)`` (differentiable in the quantizer bit-widths) to
    the model losses on every training call, so assigning beta = scale*lambda
    IS the primal MDMM term.  The dual update reads the non-trainable
    ``_ebops`` mirror variables (one step stale -- standard for
    MDMM-on-metrics; they are 0 until the first training batch, during which
    the update is skipped).  Lambda is hard-clamped at 0 (max_hard-style
    dual) since beta must be non-negative.

    NOTE: this REPLACES the static beta0 from model_config -- do not scan
    beta by hand and constrain EBOPs in the same training.

    ebops_fn: optional callable(model) -> float overriding the default
    sum-of-layer-ebops.  This is the documented adapter point for external
    resource estimators (e.g. da4ml/hls4ml post-synthesis cost models).
    """

    def __init__(self, budget: float, *, lambda_lr: float = 1e-6, damping: float = 0.0,
                 scale: float = 1.0, relative: bool = True, ebops_fn=None,
                 update_every: int = 1, name: str = "ebops"):
        super().__init__()
        if budget <= 0:
            raise ValueError("EBOPs budget must be > 0")
        self.budget = float(budget)
        self.lambda_lr = float(lambda_lr)
        self.damping = float(damping)
        self.scale = float(scale)
        self.relative = bool(relative)
        self.ebops_fn = ebops_fn
        self.update_every = int(update_every)
        self.name = name
        self.lmbda = 0.0
        self.last_ebops = 0.0
        self.last_inf = 0.0
        self._step = 0

    def total_ebops(self) -> float:
        if self.ebops_fn is not None:
            return float(self.ebops_fn(self.model))
        total = 0.0
        for layer in self.model._flatten_layers():
            e = getattr(layer, "_ebops", None)
            if e is not None:
                total += float(ops.convert_to_numpy(e))
        return total

    def _set_beta(self, beta: float):
        for layer in self.model._flatten_layers():
            b = getattr(layer, "_beta", None)
            if b is not None:
                b.assign(ops.cast(beta, b.dtype))

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1
        if self._step % self.update_every:
            return
        ebops = self.total_ebops()
        if ebops <= 0.0 and self.ebops_fn is None:
            return  # ebops mirrors not populated yet (first batch)
        g = ebops / self.budget - 1.0 if self.relative else ebops - self.budget
        self.lmbda = max(0.0, self.lmbda + self.lambda_lr * self.update_every * g)
        self._set_beta(self.scale * (self.lmbda + self.damping * max(g, 0.0)))
        self.last_ebops = ebops
        self.last_inf = g

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs[f"lambda_{self.name}"] = self.lmbda
            logs[f"{self.name}_total"] = self.last_ebops
            logs[f"inf_{self.name}"] = self.last_inf


# ---------------------------------------------------------------------------
# Named constraint targets (config-driven; docs/constrained-training.md)
# ---------------------------------------------------------------------------

def make_pt_bias_fn(head: str = "pT_output"):
    """Per-batch differentiable proxy for the pt-regression response bias.

    The head regresses the ratio r = pt_gen / pt_reco (clipped to [0.3, 2] in
    labels.py, so the denominator below is >= 0.3 and safe), hence
    pred/true = pt_pred_phys / pt_gen is the response and

        fn = | mean_batch( pred / true - 1 ) |

    is a stochastic estimate of the global relative bias; at the default
    batch size (2048) the estimator noise is ~sigma_response/45, well below
    the 1% constraint scale.  Use with a MaxConstraint at 0.01 for the
    study's "bias <= 1%" target.
    """

    def fn(ctx: ConstraintContext):
        pred = ops.reshape(ctx.y_pred[head], (-1,))
        true = ops.reshape(ops.cast(ctx.y_true[head], pred.dtype), (-1,))
        return ops.abs(ops.mean(pred / true - 1.0))

    return fn


_HEAD_LOSSES = {
    "categorical_crossentropy": lambda yt, yp, from_logits: ops.mean(
        keras.losses.categorical_crossentropy(yt, yp, from_logits=from_logits)
    ),
    "mse": lambda yt, yp, _: ops.mean(ops.square(yp - yt)),
    "mae": lambda yt, yp, _: ops.mean(ops.abs(yp - yt)),
    "logcosh": lambda yt, yp, _: ops.mean(keras.losses.log_cosh(yt, yp)),
}


def make_head_loss_fn(head: str, loss: str = "categorical_crossentropy", from_logits: bool = True):
    """Mean per-batch loss of one named head -- the study's aux-head accuracy
    floor ("aux CE <= eps instead of a weighted sum"); typically combined with
    loss_weight 0 for that head so the bound alone drives it."""
    if loss not in _HEAD_LOSSES:
        raise ValueError(f"unknown head_loss loss '{loss}'; known: {sorted(_HEAD_LOSSES)}")
    lfn = _HEAD_LOSSES[loss]

    def fn(ctx: ConstraintContext):
        yp = ctx.y_pred[head]
        yt = ops.cast(ctx.y_true[head], yp.dtype)
        if len(ops.shape(yp)) > 1 and ops.shape(yp)[-1] == 1:
            yp = ops.reshape(yp, (-1,))
            yt = ops.reshape(yt, (-1,))
        return lfn(yt, yp, from_logits)

    return fn


def make_soft_efficiency_fn(head: str, select_classes, score_classes, class_labels,
                            threshold: float = 0.5, temperature: float = 0.05):
    """Soft per-batch efficiency at a fixed working point (the study's
    rate-proxy formulation: "loss on a fixed threshold quantile").

        s    = sum of softmax probs over score_classes (the WP discriminant)
        fn   = mean over jets whose TRUE class is in select_classes of
               sigmoid((s - threshold) / temperature)

    With select_classes = background classes this is the soft background
    efficiency (rate proxy) -> MaxConstraint; with select_classes = signal
    classes it is the soft signal efficiency -> MinConstraint (efficiency
    floor).  temperature -> 0 recovers the hard counting efficiency; the
    default 0.05 keeps usable gradients.  Batches with no selected jets
    contribute ~0 via the eps-guarded denominator.
    """
    sel_idx = [class_labels.index(c) for c in select_classes]
    score_idx = [class_labels.index(c) for c in score_classes]

    def fn(ctx: ConstraintContext):
        logits = ctx.y_pred[head]
        p = ops.softmax(logits, axis=-1)
        s = ops.sum(ops.take(p, score_idx, axis=-1), axis=-1)
        soft = ops.sigmoid((s - threshold) / temperature)
        yt = ops.cast(ctx.y_true[head], soft.dtype)
        m = ops.sum(ops.take(yt, sel_idx, axis=-1), axis=-1)
        return ops.sum(soft * m) / (ops.sum(m) + 1e-7)

    return fn


_CONSTRAINT_TYPES = {
    "eq": EqConstraint,
    "max": MaxConstraint,
    "min": MinConstraint,
    "max_hard": MaxConstraintHard,
    "min_hard": MinConstraintHard,
}


def build_constraints(cons_cfg: dict, output_names, class_labels=None):
    """Translate the yaml ``constraints`` section into
    ([loss-level Constraint], [Callback]).

    Schema (see configs/deepset_hgq2_mdmm.yaml and
    docs/constrained-training.md):

        constraints:
          damping: 1.0            # default for all items
          items:
            - target: pt_bias | head_loss | bkg_eff | sig_eff | ebops
              type: eq | max | min | max_hard | min_hard   (default max)
              value: <bound / budget>
              scale / damping / name: optional per-item
              # target-specific: head, loss, from_logits, signal_classes,
              # background_classes, threshold, temperature, lambda_lr,
              # relative, update_every
    """
    items = cons_cfg.get("items") or []
    if not items:
        raise ValueError("constraints section present but 'items' is empty")
    default_damping = float(cons_cfg.get("damping", 1.0))
    constraints, callbacks, used_names = [], [], set()

    for i, item in enumerate(items):
        target = item.get("target")
        name = item.get("name", target if target not in used_names else f"{target}_{i}")
        used_names.add(name)

        if target == "ebops":
            callbacks.append(EBOPsConstraint(
                budget=item["value"],
                lambda_lr=item.get("lambda_lr", 1e-6),
                damping=item.get("damping", 0.0),
                scale=item.get("scale", 1.0),
                relative=item.get("relative", True),
                update_every=item.get("update_every", 1),
                name=name,
            ))
            continue

        if target == "pt_bias":
            fn = make_pt_bias_fn(item.get("head", "pT_output"))
        elif target == "head_loss":
            fn = make_head_loss_fn(item["head"], item.get("loss", "categorical_crossentropy"),
                                   item.get("from_logits", True))
        elif target in ("bkg_eff", "sig_eff"):
            if class_labels is None:
                raise ValueError(f"'{target}' constraint needs class_labels")
            select = item["background_classes"] if target == "bkg_eff" else item["signal_classes"]
            fn = make_soft_efficiency_fn(
                item.get("head", "jet_id_output"), select, item["score_classes"],
                class_labels, item.get("threshold", 0.5), item.get("temperature", 0.05),
            )
        elif callable(item.get("fn")):
            fn = item["fn"]  # programmatic escape hatch (tests / notebooks)
        else:
            raise ValueError(f"unknown constraint target '{target}'")

        ctype = item.get("type", "max")
        if ctype not in _CONSTRAINT_TYPES:
            raise ValueError(f"unknown constraint type '{ctype}'; known: {sorted(_CONSTRAINT_TYPES)}")
        kw = dict(scale=item.get("scale", 1.0), damping=item.get("damping", default_damping),
                  name=name)
        constraints.append(_CONSTRAINT_TYPES[ctype](fn, item["value"], **kw))

    return constraints, callbacks


def attach_mdmm(base_model: keras.Model, cons_cfg: dict, output_names, class_labels=None):
    """Build the training model + callbacks for a ``constraints`` config.

    Returns (train_model, callbacks): train_model is an MDMMModel wrapper when
    loss-level constraints are present, else the base model unchanged (an
    ebops-only config needs no wrapper -- the callback is self-contained).
    """
    constraints, callbacks = build_constraints(cons_cfg, output_names, class_labels)
    if not constraints:
        return base_model, callbacks
    wrapper = MDMMModel(base_model, constraints, output_names)
    return wrapper, [MDMMLogger(wrapper)] + callbacks


def collect_constraint_metrics(train_model, callbacks) -> dict:
    """Final multiplier/infeasibility values for mlflow logging."""
    out = {}
    if isinstance(train_model, MDMMModel):
        for c in train_model.mdmm_constraints:
            out[f"lambda_{c.name}"] = float(ops.convert_to_numpy(c.lmbda))
            out[f"inf_{c.name}"] = float(ops.convert_to_numpy(c._last_inf))
    for cb in callbacks or []:
        if isinstance(cb, EBOPsConstraint):
            out[f"lambda_{cb.name}"] = cb.lmbda
            out[f"{cb.name}_total"] = cb.last_ebops
            out[f"inf_{cb.name}"] = cb.last_inf
    return out
