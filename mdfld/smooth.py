"""Temporal smoothing for video: the One Euro filter (Casiez, Roussel and Vogel, CHI 2012).

It is a low-pass filter whose cut-off rises with speed: slow jitter is removed, while fast motion such as a blink
passes with little lag. Coordinates are divided by a fixed scale (the face's eye width) first, so the parameters do
not depend on image size. For a whole video, smooth_sequence runs it forwards and backwards and averages the two,
which cancels the lag.
"""
import math

import numpy as np


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """Filters an array of any shape, one call per frame."""

    def __init__(self, fps, min_cutoff=1.0, beta=0.5, d_cutoff=1.0, scale=1.0):
        self.dt, self.min_cutoff, self.beta, self.d_cutoff, self.scale = 1.0 / fps, min_cutoff, beta, d_cutoff, scale
        self.x = self.dx = None

    def reset(self):
        self.x = self.dx = None

    def __call__(self, x):
        x = np.asarray(x, np.float64) / self.scale
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx = x, np.zeros_like(x)
            return x * self.scale
        dx = (x - self.x) / self.dt
        a_d = _alpha(self.d_cutoff, self.dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + 1.0 / (2 * math.pi * cutoff * self.dt))
        self.x = a * x + (1 - a) * self.x
        return self.x * self.scale


def smooth_sequence(X, fps, min_cutoff=1.0, beta=5.0, d_cutoff=2.0, scale=1.0):
    """Zero-lag smoothing of a (T, ...) array: One Euro forwards and backwards, averaged."""
    X = np.asarray(X, np.float64)
    if len(X) < 3:
        return X

    def run(seq):
        f = OneEuro(fps, min_cutoff, beta, d_cutoff, scale)
        return np.stack([f(x) for x in seq])
    return 0.5 * (run(X) + run(X[::-1])[::-1])
