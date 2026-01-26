# slbt/reporting.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import Node


@dataclass
class NodeRecord:
    """Single row of the tree report."""
    id: int
    node_type: str          # "Leaf" or "Internal"
    feature: Optional[str]  # splitting feature (None for leaf)
    threshold: Any          # raw threshold info (stringified)
    N: int                  # number of samples at node
    impurity: float
    distribution: Any       # raw distribution (list of floats)
    labels: Any             # label values (list)
    gpi: Optional[float]
    pi: Optional[float]
    gcr: Any                # None or list
    lift_left: Any          # None or list/str
    lift_right: Any         # None or list/str


class TreeReporter:
    """
    Collects per-node information into a pandas DataFrame while the tree grows.
    """

    def __init__(self, homogeneity: Optional[str] = None, decimals: int = 4):
        self.homogeneity = homogeneity
        self.decimals = decimals
        self._records: List[NodeRecord] = []

    @property
    def results(self) -> pd.DataFrame:
        """Return the accumulated report as a pandas DataFrame."""
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "id", "node_type", "feature", "threshold", "N",
                    "impurity", "distribution", "labels",
                    "gpi", "pi", "gcr", "lift_left", "lift_right",
                ]
            )
        return pd.DataFrame(asdict(r) for r in self._records)

    # ---------- public API ----------
    def add_node(self, node: Node, is_leaf: bool) -> None:
        node_type = "Leaf" if is_leaf else "Internal"

        # distribution e labels
        dist_raw = (
            node.distribution.tolist()
            if isinstance(node.distribution, np.ndarray)
            else node.distribution
        )
        dist = self._round_nested(dist_raw)

        labels = (
            node.labels.tolist()
            if isinstance(node.labels, np.ndarray)
            else node.labels
        )

        # GCR
        gcr_raw = getattr(node, "GCR", None)
        if isinstance(gcr_raw, np.ndarray):
            gcr_raw = gcr_raw.tolist()
        gcr = self._round_nested(gcr_raw)

        # LIFT
        lift_left_raw, lift_right_raw = self._serialize_lift(node)
        lift_left = self._round_nested(lift_left_raw)
        lift_right = self._round_nested(lift_right_raw)

        # threshold (non necessariamente numerico, quindi non arrotondo qui)
        threshold = self._serialize_threshold(node)

        rec = NodeRecord(
            id=node.position,
            node_type=node_type,
            feature=node.feature if not is_leaf else None,
            threshold=threshold,
            N=node.N,
            impurity=float(node.impurity) if node.impurity is not None else None,
            distribution=dist,
            labels=labels,
            gpi=float(node.gpi) if node.gpi is not None else None,
            pi=float(node.pi) if node.pi is not None else None,
            gcr=gcr,
            lift_left=lift_left,
            lift_right=lift_right,
        )
        
        self._records.append(rec)

    # ---------- helpers ----------
    def _serialize_threshold(self, node: Node) -> Any:
        """
        Convert node.treshold into a JSON/pandas-friendly format.
        """
        thr = node.treshold

        if thr is None:
            return None

        # without strat_labels: flat list
        if getattr(node, "strat_labels", None) is None:
            # es. np.ndarray di valori
            if isinstance(thr, (np.ndarray, list, tuple)):
                return list(thr)
            return thr

        # with strat_labels: mapping stratum → values
        strat_labels = list(node.strat_labels)
        try:
            return {
                str(s): list(v) for s, v in zip(strat_labels, thr)
            }
        except TypeError:
            # fallback grezzo
            return str(thr)

    def _serialize_lift(self, node: Node) -> (Any, Any):
        """
        Convert LIFT_1 and LIFT_2 to something storable.
        """
        L1 = getattr(node, "LIFT_1", None)
        L2 = getattr(node, "LIFT_2", None)

        def to_python(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                # lista di array → lista di liste
                return [
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in x
                ]
            return x

        return to_python(L1), to_python(L2)

    def _round_nested(self, x: Any) -> Any:
        """
        - float/int -> rounded float
        - np.ndarray -> list of rounded floats
        - lista/tupla -> same structure, rounded values
        - dict -> same dictionary, rounded values
        - altro -> gets back unchanged
        """
        d = self.decimals

        if x is None:
            return None

        # singolo numero
        if isinstance(x, (float, int)):
            return round(float(x), d)

        # numpy array
        import numpy as np  # safe import qui
        if isinstance(x, np.ndarray):
            return [self._round_nested(v) for v in x.tolist()]

        # lista o tupla
        if isinstance(x, (list, tuple)):
            return [self._round_nested(v) for v in x]

        # dict
        if isinstance(x, dict):
            return {k: self._round_nested(v) for k, v in x.items()}

        # qualsiasi altro tipo (stringhe, ecc.)
        return x