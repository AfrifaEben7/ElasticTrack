"""
elastic_bandit.py — LinUCB Contextual Bandit for online elastic ratio adaptation.

Extends the PE-gating framework from elastic_xlstm_paper2 (PEThresholdBandit)
to adapt SAM2's ElasticController threshold ratios online during streaming
inference — requiring no ground truth.

Instead of adapting two PE thresholds (theta_H / theta_L) as in the IDS domain,
we adapt three ratio multipliers that the ElasticController uses to derive
absolute thresholds from its self-calibrated mean delta:
  ECO_RATIO   : delta < mean * ECO_RATIO   → ECO  (N_MEM=2)
  BOOST_RATIO : delta > mean * BOOST_RATIO → BOOST (N_MEM=7)
  (SKIP_RATIO is left fixed; rarely triggered)

State vector (8-dim, all in [0, 1]):
  [pe_ema, delta_ema_norm, iou_ema, churn_ema,
   eco_ratio_norm, boost_ratio_norm, skip_ratio_norm, n_mem_norm]

Actions (5 discrete):
  0: no change
  1: tighten  (eco +0.05, boost +0.20) → save more compute, accept less BOOST
  2: loosen   (eco -0.05, boost -0.20) → spend more compute, allow more BOOST
  3: widen    (boost +0.20)            → raise BOOST threshold (harder to trigger)
  4: narrow   (eco -0.05)              → lower ECO threshold (more ECO frames)

Reward (no GT required):
  R = iou_ema - CHURN_PENALTY * churn_ema - COMPUTE_PENALTY * n_mem_norm
  Optimises: segmentation quality + track stability + compute efficiency.

Usage:
    bandit = ElasticBandit(alpha=0.5, forget=0.99, update_interval=5)
    # Each frame:
    bandit.observe(mean_iou, track_churn)
    eco_r, boost_r, skip_r = bandit.step(elastic_controller)
    elastic_controller.update_ratios(eco_r, boost_r, skip_r)

Cross-paper contribution:
    This is the first application of a LinUCB contextual bandit to adapt memory
    depth in a visual foundation model, extending the elastic_xlstm framework
    from time-series (IDS V2X) to streaming video segmentation.
"""

import numpy as np
from typing import Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────

D_STATE   = 8
N_ACTIONS = 5

ECO_RATIO_MIN,   ECO_RATIO_MAX   = 0.20, 0.80
BOOST_RATIO_MIN, BOOST_RATIO_MAX = 0.80, 4.00
SKIP_RATIO_MIN,  SKIP_RATIO_MAX  = 0.02, 0.25

ECO_STEP   = 0.05
BOOST_STEP = 0.20

# (d_eco, d_boost): adjustments applied to ratio multipliers
ACTIONS = {
    0: ( 0.0,       0.0),       # hold
    1: (+ECO_STEP, +BOOST_STEP),  # tighten: less BOOST/ECO → save compute
    2: (-ECO_STEP, -BOOST_STEP),  # loosen : more BOOST → better on dynamic scenes
    3: ( 0.0,      +BOOST_STEP),  # widen  : raise BOOST threshold only
    4: (-ECO_STEP,  0.0),         # narrow : lower ECO threshold only
}

CHURN_PENALTY   = 0.40   # weight on track instability in reward
COMPUTE_PENALTY = 0.15   # weight on N_MEM usage in reward (efficiency incentive)


class ElasticBandit:
    """
    LinUCB contextual bandit for online PE threshold ratio adaptation.

    Parameters
    ----------
    alpha : float
        Exploration coefficient (UCB width). 0.3–1.0 typical range.
        Lower → exploit faster; higher → explore longer.
    forget : float
        Sliding-window forgetting factor in (0, 1). Applied as
        A ← forget*A + x x^T so older observations decay exponentially.
        0.99 gives ~100-step effective window.
    update_interval : int
        Frames between bandit decisions. 5–10 reduces noise in the
        reward signal without adding significant latency.
    eco_ratio_init, boost_ratio_init, skip_ratio_init : float
        Starting ratio values — these match ElasticController defaults.
    """

    def __init__(
        self,
        alpha: float = 0.50,
        forget: float = 0.99,
        update_interval: int = 5,
        eco_ratio_init:   float = 0.45,
        boost_ratio_init: float = 2.00,
        skip_ratio_init:  float = 0.10,
    ):
        self.alpha           = alpha
        self.forget          = forget
        self.update_interval = update_interval

        self.eco_ratio   = eco_ratio_init
        self.boost_ratio = boost_ratio_init
        self.skip_ratio  = skip_ratio_init

        # Per-action ridge regression (LinUCB)
        self.A = [np.eye(D_STATE, dtype=np.float64) for _ in range(N_ACTIONS)]
        self.b = [np.zeros(D_STATE, dtype=np.float64) for _ in range(N_ACTIONS)]

        self._last_action = None
        self._last_state  = None
        self._frame       = 0

        # Streaming EMA accumulators for reward signals
        self._iou_ema   = 0.50   # tracks segmentation quality
        self._churn_ema = 0.05   # tracks birth+death rate per frame

        # History for diagnostics
        self._action_counts = [0] * N_ACTIONS
        self._reward_history: list = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def observe(self, mean_iou: float, track_churn: float) -> None:
        """
        Update EMA reward accumulators. Call every frame (before step).

        Parameters
        ----------
        mean_iou : float
            Mean predicted IoU across currently active SAM2 tracks [0, 1].
            Higher = better segmentation quality.
        track_churn : float
            (births + deaths) / active_tracks this frame [0, inf].
            Lower = more stable tracking.
        """
        self._iou_ema   = 0.30 * mean_iou   + 0.70 * self._iou_ema
        self._churn_ema = 0.30 * min(track_churn, 2.0) + 0.70 * self._churn_ema

    def step(self, elastic) -> Tuple[float, float, float]:
        """
        One bandit step: compute reward, update matrices, select action, apply.
        Only active every update_interval frames; otherwise returns current ratios.

        Parameters
        ----------
        elastic : ElasticController
            The live controller — bandit reads its internal state.

        Returns
        -------
        (eco_ratio, boost_ratio, skip_ratio) : Tuple[float, float, float]
            Updated ratio multipliers to pass to elastic.update_ratios().
        """
        self._frame += 1
        if self._frame % self.update_interval != 0:
            return self.eco_ratio, self.boost_ratio, self.skip_ratio

        # Reward from previous action window
        r = (self._iou_ema
             - CHURN_PENALTY   * self._churn_ema
             - COMPUTE_PENALTY * (elastic._n_mem / 7.0))
        self._reward_history.append(round(r, 4))

        # Build state vector
        s = self._build_state(elastic)

        # Update matrices for previous action
        if self._last_action is not None:
            a = self._last_action
            ps = self._last_state
            self.A[a] = self.forget * self.A[a] + np.outer(ps, ps)
            self.b[a] = self.forget * self.b[a] + r * ps

        # Select action via UCB
        scores = []
        for a in range(N_ACTIONS):
            A_inv = np.linalg.inv(self.A[a])
            theta_hat = A_inv @ self.b[a]
            ucb = theta_hat @ s + self.alpha * np.sqrt(float(s @ A_inv @ s))
            scores.append((ucb, a))
        action = max(scores, key=lambda x: x[0])[1]

        self._last_action = action
        self._last_state  = s
        self._action_counts[action] += 1

        # Apply action: adjust ECO and BOOST ratios
        d_eco, d_boost = ACTIONS[action]
        self.eco_ratio   = float(np.clip(
            self.eco_ratio   + d_eco,   ECO_RATIO_MIN,   ECO_RATIO_MAX))
        self.boost_ratio = float(np.clip(
            self.boost_ratio + d_boost, BOOST_RATIO_MIN, BOOST_RATIO_MAX))

        # Enforce minimum separation (eco must be comfortably below boost)
        if self.eco_ratio >= self.boost_ratio * 0.65:
            self.eco_ratio = round(self.boost_ratio * 0.60, 3)
            self.eco_ratio = max(self.eco_ratio, ECO_RATIO_MIN)

        return self.eco_ratio, self.boost_ratio, self.skip_ratio

    def stats(self) -> dict:
        recent = self._reward_history[-20:] if self._reward_history else []
        return {
            "eco_ratio":      round(self.eco_ratio,   3),
            "boost_ratio":    round(self.boost_ratio, 3),
            "skip_ratio":     round(self.skip_ratio,  3),
            "iou_ema":        round(self._iou_ema,    3),
            "churn_ema":      round(self._churn_ema,  3),
            "mean_reward":    round(float(np.mean(recent)), 4) if recent else None,
            "action_counts":  dict(enumerate(self._action_counts)),
            "steps":          self._frame // self.update_interval,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_state(self, elastic) -> np.ndarray:
        """Construct 8-dim normalised state vector from ElasticController internals."""
        pe_ema    = float(elastic._ema_pe)    if elastic._ema_pe    is not None else 0.0
        delta_ema = float(elastic._ema_delta) if elastic._ema_delta is not None else 0.0
        calib     = float(elastic._calib_mean) if elastic._calib_mean is not None else 1.0

        return np.array([
            np.clip(pe_ema, 0, 1),
            np.clip(delta_ema / max(calib, 1e-8), 0, 3) / 3.0,  # normalise [0,1]
            np.clip(self._iou_ema, 0, 1),
            np.clip(self._churn_ema / 2.0, 0, 1),                # cap at 2.0
            (self.eco_ratio   - ECO_RATIO_MIN)   / (ECO_RATIO_MAX   - ECO_RATIO_MIN),
            (self.boost_ratio - BOOST_RATIO_MIN) / (BOOST_RATIO_MAX - BOOST_RATIO_MIN),
            self.skip_ratio / SKIP_RATIO_MAX,
            float(elastic._n_mem) / 7.0,
        ], dtype=np.float64)
