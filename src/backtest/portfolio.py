import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class PortfolioConfig:
    weights: Dict[str, float]
    initial_capital: float = 10000.0


def validate_weights(weights: Dict[str, float]) -> None:
    """Verifica che i pesi siano validi (>=0 e somma ≈ 1)."""
    if not all(0 <= w <= 1 for w in weights.values()):
        raise ValueError("I pesi devono essere compresi tra 0 e 1.")
    if not np.isclose(sum(weights.values()), 1):
        raise ValueError("La somma dei pesi deve essere uguale a 1.")
    pass


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalizza i pesi in modo che la somma sia 1."""
    total = sum(weights.values())
    return {symbol: weight / total for symbol, weight in weights.items()}


def _build_weights(symbols: list[str], weights: Optional[list[float]]) -> dict:
    if weights is None:
        # Default: pesi uguali se non specificati.
        equal_weight = 1.0 / len(symbols)
        weights = [equal_weight] * len(symbols)
    if len(weights) != len(symbols):
        raise ValueError("Il numero di pesi deve corrispondere al numero di simboli.")
    return {sym: float(w) for sym, w in zip(symbols, weights)}