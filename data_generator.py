"""
data_generator.py
==================
Generador de datos SINTETICOS para el caso de tostado de cacao Nacional.

IMPORTANTE (declaracion de transparencia, va tambien en el reporte):
No se usaron datos experimentales reales. Los valores se generaron con
funciones polinomicas cuyos SIGNOS y TENDENCIAS estan basados en patrones
reportados en literatura cientifica sobre tostado de cacao:

  - Los polifenoles totales y la actividad antioxidante (DPPH) DISMINUYEN
    con temperatura y tiempo de tostado mas altos (degradacion termica de
    compuestos fenolicos).
  - El indice de pardeamiento (color) AUMENTA con temperatura y tiempo
    (reacciones de Maillard y caramelizacion).
  - El puntaje sensorial tiene un OPTIMO INTERMEDIO: sub-tostado sabe
    astringente/crudo, sobre-tostado sabe quemado/amargo.
  - La humedad inicial del grano modula la velocidad de transferencia de
    calor y por tanto interactua con temperatura y tiempo.

Se anade ruido aleatorio (gaussiano) para simular variabilidad experimental
realista. Los coeficientes fueron calibrados para que el optimo resultante
caiga cerca del rango reportado en estudios reales (~127-130 C, ~10-17 min),
lo cual sirve como validacion de sentido fisico del ejercicio, NO como
prueba de exactitud experimental.
"""

import numpy as np
import pandas as pd
from rsm_core import generate_ccd, design_to_real, coded_to_real

# Rangos reales de los 3 factores del proceso
FACTOR_RANGES = {
    "Temperatura": (120, 160),   # C
    "Tiempo": (10, 30),          # min
    "Humedad": (5, 8),           # % base humeda
}

FACTOR_NAMES = list(FACTOR_RANGES.keys())


def _coded_cols(df_real: pd.DataFrame) -> pd.DataFrame:
    """Recupera las columnas codificadas a partir de las reales (uso interno)."""
    out = pd.DataFrame(index=df_real.index)
    for f, (lo, hi) in FACTOR_RANGES.items():
        center = (hi + lo) / 2
        half = (hi - lo) / 2
        out[f] = (df_real[f] - center) / half
    return out


def simulate_responses(df_coded: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Recibe un dataframe con columnas Temperatura, Tiempo, Humedad EN UNIDADES
    CODIFICADAS [-alpha, alpha] y devuelve las 4 respuestas simuladas.
    """
    rng = np.random.default_rng(seed)
    T = df_coded["Temperatura"].values
    t = df_coded["Tiempo"].values
    H = df_coded["Humedad"].values
    n = len(df_coded)

    # --- Polifenoles totales (mg GAE/g) ---
    # Base ~24 mg GAE/g en el centro, cae con T y t, leve interaccion con H
    polifenoles = (
        24.0
        - 3.2 * T
        - 2.1 * t
        - 0.6 * T ** 2
        - 0.4 * t ** 2
        - 0.5 * T * t
        + 0.5 * H
        + rng.normal(0, 0.5, n)
    )
    polifenoles = np.clip(polifenoles, 5, None)

    # --- Actividad antioxidante DPPH (% inhibicion) ---
    dpph = (
        68.0
        - 5.5 * T
        - 3.0 * t
        - 1.0 * T ** 2
        - 0.5 * t ** 2
        - 0.8 * T * t
        + 0.8 * H
        + rng.normal(0, 1.0, n)
    )
    dpph = np.clip(dpph, 10, 95)

    # --- Indice de pardeamiento (color, adimensional 0-100) ---
    pardeamiento = (
        45.0
        + 8.5 * T
        + 6.0 * t
        + 1.2 * T ** 2
        + 0.8 * t ** 2
        + 1.5 * T * t
        - 1.0 * H
        + rng.normal(0, 1.2, n)
    )
    pardeamiento = np.clip(pardeamiento, 10, 100)

    # --- Puntaje sensorial (escala 1-9) con optimo intermedio ---
    # Maximo cerca de T=-0.2 (~128 C) y t=-0.4 (~14 min) en unidades codificadas
    sensorial = (
        7.6
        - 1.4 * (T + 0.2) ** 2
        - 1.1 * (t + 0.4) ** 2
        - 0.3 * T * t
        + 0.15 * H
        + rng.normal(0, 0.2, n)
    )
    sensorial = np.clip(sensorial, 1, 9)

    out = df_coded.copy()
    out["Polifenoles_mgGAE_g"] = np.round(polifenoles, 2)
    out["DPPH_pct_inhibicion"] = np.round(dpph, 2)
    out["Indice_pardeamiento"] = np.round(pardeamiento, 2)
    out["Puntaje_sensorial"] = np.round(sensorial, 2)
    return out


def generate_dataset(n_center: int = 6, alpha: str = "rotatable", seed: int = 42) -> pd.DataFrame:
    """
    Genera el dataset completo: diseno CCD (20 corridas para k=3) +
    respuestas simuladas, con columnas en unidades REALES y codificadas.
    """
    design_coded = generate_ccd(k=3, n_center=n_center, alpha=alpha,
                                 factor_names=FACTOR_NAMES)
    responses = simulate_responses(design_coded[FACTOR_NAMES], seed=seed)

    design_real = design_to_real(design_coded[FACTOR_NAMES], FACTOR_RANGES)

    final = design_coded[["Run", "PointType"]].copy()
    for f in FACTOR_NAMES:
        final[f"{f}_real"] = design_real[f].round(2)
        final[f"{f}_cod"] = design_coded[f].round(3)
    for col in ["Polifenoles_mgGAE_g", "DPPH_pct_inhibicion",
                "Indice_pardeamiento", "Puntaje_sensorial"]:
        final[col] = responses[col]

    return final


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/cacao_ccd_sintetico.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nGenerado dataset con {len(df)} corridas -> data/cacao_ccd_sintetico.csv")
