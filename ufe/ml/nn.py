"""A small multi-task neural network in plain NumPy (no PyTorch needed, so it trains anywhere:
a laptop, GitHub Actions). Written out step by step so every part can be read:

    inputs (standardised features)
      -> dense 32, ReLU -> dense 16, ReLU            ("hidden layers": learned combinations)
      -> three outputs:
           start logit          P(start) = sigmoid(output 1)
           log goals per 90     goals ~ Poisson(exp(output 2) * minutes / 90)
           log assists per 90   assists ~ Poisson(exp(output 3) * minutes / 90)

Learning = minimising the loss (how wrong the predictions were on past matches) by gradient
descent: the forward pass makes predictions, backpropagation gives each weight's share of the
error, and Adam nudges every weight against it, a mini-batch at a time. Training stops when the
loss on a held-out season stops improving (early stopping), which is what keeps the network from
memorising noise. With hidden=() the same class is a plain logistic + Poisson regression, the
baseline the network has to beat.
"""
from __future__ import annotations

import json

import numpy as np

CLIP = 12.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -CLIP, CLIP)))


def _softplus(z):
    return np.logaddexp(0.0, z)


class MultiTaskNet:
    def __init__(self, n_in: int, hidden: tuple = (32, 16), l2: float = 1e-4, seed: int = 0,
                 base: tuple = (0.28, 0.12, 0.08), weights: tuple = (1.0, 1.0, 1.0)):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 3]
        self.hidden, self.l2, self.task_w = tuple(hidden), float(l2), np.array(weights, dtype=float)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)) * (0.1 if i == len(sizes) - 2 else 1.0)
                  for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
        self.b = [np.zeros(b) for b in sizes[1:]]
        # start the outputs at the base rates, so early training refines rather than discovers them
        self.b[-1] = np.array([np.log(base[0] / (1 - base[0])), np.log(base[1]), np.log(base[2])])
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)
        self.history: list = []

    # ------------------------------------------------------------------ forward and backward
    def _forward(self, X):
        hs, zs = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            zs.append(z)
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            hs.append(h)
        return h, hs, zs

    def loss_and_grads(self, X, T):
        """T: dict of arrays start (0/1), played (0/1), expo (minutes/90), goals, assists."""
        out, hs, zs = self._forward(X)
        n = len(X)
        npl = max(float(T["played"].sum()), 1.0)
        z_s, z_g, z_a = out[:, 0], np.clip(out[:, 1], -CLIP, 5), np.clip(out[:, 2], -CLIP, 5)
        p = _sigmoid(z_s)
        l_s = float(np.mean(_softplus(z_s) - T["start"] * z_s))
        mu_g, mu_a = np.exp(z_g) * T["expo"], np.exp(z_a) * T["expo"]
        pl = T["played"]
        l_g = float(np.sum(pl * (mu_g - T["goals"] * np.log(mu_g + 1e-12))) / npl)
        l_a = float(np.sum(pl * (mu_a - T["assists"] * np.log(mu_a + 1e-12))) / npl)
        w = self.task_w
        loss = w[0] * l_s + w[1] * l_g + w[2] * l_a + 0.5 * self.l2 * sum(float((W ** 2).sum()) for W in self.W)
        d = np.stack([w[0] * (p - T["start"]) / n, w[1] * pl * (mu_g - T["goals"]) / npl,
                      w[2] * pl * (mu_a - T["assists"]) / npl], axis=1)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = hs[i].T @ d + self.l2 * self.W[i]
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (zs[i - 1] > 0)
        return loss, (l_s, l_g, l_a), gW, gb

    # ------------------------------------------------------------------ training
    def fit(self, X, T, Xv, Tv, epochs: int = 80, batch: int = 512, lr: float = 2e-3, patience: int = 8,
            seed: int = 0, verbose: bool = False) -> "MultiTaskNet":
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        Xs, Xvs = (X - self.mu) / self.sd, (Xv - self.mu) / self.sd
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(q) for q in params]
        v = [np.zeros_like(q) for q in params]
        b1, b2, step = 0.9, 0.999, 0
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            idx = rng.permutation(len(Xs))
            for k in range(0, len(idx), batch):
                j = idx[k:k + batch]
                _, _, gW, gb = self.loss_and_grads(Xs[j], {c: a[j] for c, a in T.items()})
                step += 1
                for q, g, mq, vq in zip(params, gW + gb, m, v):
                    mq *= b1
                    mq += (1 - b1) * g
                    vq *= b2
                    vq += (1 - b2) * g * g
                    q -= lr * (mq / (1 - b1 ** step)) / (np.sqrt(vq / (1 - b2 ** step)) + 1e-8)
            tr = self.loss_and_grads(Xs[:20000], {c: a[:20000] for c, a in T.items()})[1]
            va = self.loss_and_grads(Xvs, Tv)[1]
            vl = float(np.dot(self.task_w, va))
            self.history.append(dict(epoch=ep + 1, train=[round(x, 5) for x in tr], val=[round(x, 5) for x in va]))
            if verbose:
                print(ep + 1, [round(x, 4) for x in tr], [round(x, 4) for x in va], flush=True)
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = ([W.copy() for W in self.W], [b.copy() for b in self.b], ep + 1)
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.W, self.b, self.best_epoch = best_state[0], best_state[1], best_state[2]
        return self

    # ------------------------------------------------------------------ use
    def predict(self, X) -> dict:
        out, _, _ = self._forward((np.asarray(X, dtype=float) - self.mu) / self.sd)
        return dict(p_start=_sigmoid(out[:, 0]), g90=np.exp(np.clip(out[:, 1], -CLIP, 5)),
                    a90=np.exp(np.clip(out[:, 2], -CLIP, 5)))

    def to_json(self) -> str:
        return json.dumps(dict(hidden=list(self.hidden), l2=self.l2, task_w=self.task_w.tolist(), mu=self.mu.tolist(),
                               sd=self.sd.tolist(), W=[W.tolist() for W in self.W], b=[b.tolist() for b in self.b],
                               best_epoch=getattr(self, "best_epoch", None), history=self.history))

    @classmethod
    def from_json(cls, s: str) -> "MultiTaskNet":
        d = json.loads(s)
        W = [np.array(w) for w in d["W"]]
        net = cls(W[0].shape[0], tuple(d["hidden"]), d["l2"], weights=tuple(d["task_w"]))
        net.W, net.b = W, [np.array(b) for b in d["b"]]
        net.mu, net.sd = np.array(d["mu"]), np.array(d["sd"])
        net.best_epoch, net.history = d.get("best_epoch"), d.get("history", [])
        return net


class Ensemble:
    """Several networks trained from different random starts; predictions are averaged, which
    removes most of the run-to-run noise a small network has."""

    def __init__(self, nets: list):
        self.nets = list(nets)

    @property
    def best_epoch(self):
        eps = [getattr(n, "best_epoch", None) for n in self.nets]
        return int(round(np.mean([e for e in eps if e]))) if any(eps) else None

    def predict(self, X) -> dict:
        ps = [n.predict(X) for n in self.nets]
        return {k: np.mean([p[k] for p in ps], axis=0) for k in ps[0]}

    def to_json(self) -> str:
        return json.dumps(dict(ensemble=[json.loads(n.to_json()) for n in self.nets]))

    @classmethod
    def from_json(cls, s: str) -> "Ensemble":
        d = json.loads(s)
        if "ensemble" not in d:
            return cls([MultiTaskNet.from_json(s)])
        return cls([MultiTaskNet.from_json(json.dumps(x)) for x in d["ensemble"]])

