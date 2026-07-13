"""
rsm_core.py
============
Motor estadistico para Metodologia de Superficie de Respuesta (RSM).

Implementa:
  - Generacion de disenos: Central Compuesto (CCD) y Box-Behnken (BBD)
  - Ajuste de modelos de 1er y 2do orden (minimos cuadrados)
  - ANOVA y prueba de falta de ajuste
  - Analisis canonico (punto estacionario, eigenvalores, clasificacion)
  - Ascenso mas pronunciado
  - Analisis de cresta (ridge analysis)
  - Funcion de deseabilidad de Derringer-Suich (multi-respuesta)

Convencion: los factores se trabajan en unidades codificadas [-1, 1]
(o [-alpha, alpha] en los puntos axiales del CCD). Las funciones de
codificacion/decodificacion permiten pasar de unidades reales a codificadas
y viceversa.

Autor: Proyecto RSM - Tostado de cacao Nacional (mejorado)
"""

from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy import stats, optimize
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. CODIFICACION DE VARIABLES
# ---------------------------------------------------------------------------

def coded_to_real(x_coded: float, low: float, high: float) -> float:
    """Convierte una variable codificada [-1, 1] a su valor real."""
    center = (high + low) / 2
    half_range = (high - low) / 2
    return center + x_coded * half_range


def real_to_coded(x_real: float, low: float, high: float) -> float:
    """Convierte un valor real a su version codificada [-1, 1]."""
    center = (high + low) / 2
    half_range = (high - low) / 2
    return (x_real - center) / half_range


def design_to_real(design_coded: pd.DataFrame, factor_ranges: dict) -> pd.DataFrame:
    """
    Convierte una matriz de diseno codificada a unidades reales.
    factor_ranges: {'nombre_factor': (low, high), ...}
    """
    df = design_coded.copy()
    for col, (low, high) in factor_ranges.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda v: coded_to_real(v, low, high))
    return df


# ---------------------------------------------------------------------------
# 2. GENERACION DE DISENOS EXPERIMENTALES
# ---------------------------------------------------------------------------

def generate_ccd(k: int, n_center: int = 6, alpha: str = "rotatable",
                  factor_names: Optional[list] = None) -> pd.DataFrame:
    """
    Genera un diseno Central Compuesto (CCD) en unidades codificadas.

    Parametros
    ----------
    k : int
        Numero de factores (2 o 3 tipicamente).
    n_center : int
        Numero de puntos centrales (replicas para estimar error puro).
    alpha : str o float
        'rotatable' -> alpha = (2^k)^(1/4)
        'orthogonal' -> alpha calculado para ortogonalidad
        o un valor numerico directo.
    factor_names : list, opcional
        Nombres de los factores. Por defecto X1, X2, ...

    Retorna
    -------
    DataFrame con columnas de factores + columna 'PointType'
    ('Factorial', 'Axial', 'Center').
    """
    if factor_names is None:
        factor_names = [f"X{i+1}" for i in range(k)]
    assert len(factor_names) == k

    # Puntos factoriales: diseno factorial completo 2^k en {-1, 1}
    factorial_pts = np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)

    # Alpha
    if alpha == "rotatable":
        alpha_val = (2 ** k) ** 0.25
    elif alpha == "orthogonal":
        n_f = 2 ** k
        n_a = 2 * k
        alpha_val = ((n_f * (np.sqrt(n_f + n_a + n_center) - np.sqrt(n_f)) ** 2) / 4) ** 0.25 \
            if False else (n_f ** 0.5)  # aproximacion practica
    else:
        alpha_val = float(alpha)

    # Puntos axiales: +-alpha en cada eje, resto en 0
    axial_pts = []
    for i in range(k):
        for sign in (-1, 1):
            row = [0.0] * k
            row[i] = sign * alpha_val
            axial_pts.append(row)
    axial_pts = np.array(axial_pts)

    # Puntos centrales
    center_pts = np.zeros((n_center, k))

    df = pd.DataFrame(
        np.vstack([factorial_pts, axial_pts, center_pts]),
        columns=factor_names,
    )
    df["PointType"] = (
        ["Factorial"] * len(factorial_pts)
        + ["Axial"] * len(axial_pts)
        + ["Center"] * len(center_pts)
    )
    df.insert(0, "Run", range(1, len(df) + 1))
    df.attrs["alpha"] = alpha_val
    df.attrs["design_type"] = "CCD"
    return df.reset_index(drop=True)


def generate_bbd(k: int, n_center: int = 3,
                  factor_names: Optional[list] = None) -> pd.DataFrame:
    """
    Genera un diseno Box-Behnken (BBD) en unidades codificadas.
    Valido para k = 3, 4 o 5 factores (construccion por bloques de pares).

    Retorna
    -------
    DataFrame con columnas de factores + 'PointType' ('Edge', 'Center').
    """
    if factor_names is None:
        factor_names = [f"X{i+1}" for i in range(k)]
    assert k >= 3, "Box-Behnken requiere al menos 3 factores"

    rows = []
    # Para cada par de factores, se genera un factorial 2^2 en esos dos ejes
    # dejando los demas factores en el punto central (0)
    for i, j in itertools.combinations(range(k), 2):
        for si, sj in itertools.product([-1, 1], repeat=2):
            row = [0.0] * k
            row[i] = si
            row[j] = sj
            rows.append(row)

    edge_pts = np.array(rows)
    center_pts = np.zeros((n_center, k))

    df = pd.DataFrame(np.vstack([edge_pts, center_pts]), columns=factor_names)
    df["PointType"] = ["Edge"] * len(edge_pts) + ["Center"] * len(center_pts)
    df.insert(0, "Run", range(1, len(df) + 1))
    df.attrs["design_type"] = "BBD"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. AJUSTE DE MODELOS (1er y 2do orden)
# ---------------------------------------------------------------------------

@dataclass
class RSMModel:
    order: int
    factor_names: list
    response_name: str
    coefficients: pd.Series
    fitted_model: object  # statsmodels results object
    X_design: pd.DataFrame
    y: pd.Series
    r2: float
    r2_adj: float

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        Xb = _build_design_matrix(X_new, self.factor_names, self.order)
        pred = self.fitted_model.predict(Xb)
        # En algunas versiones de statsmodels/pandas, predict() devuelve una
        # pandas.Series en vez de un numpy.ndarray. Se normaliza siempre a
        # ndarray para que .reshape(), indexado posicional, etc. funcionen
        # igual en cualquier entorno.
        return np.asarray(pred)


def _build_design_matrix(df: pd.DataFrame, factor_names: list, order: int) -> pd.DataFrame:
    """Construye la matriz de diseno (con intercepto) para orden 1 o 2."""
    import statsmodels.api as sm

    X = pd.DataFrame(index=df.index)
    for f in factor_names:
        X[f] = df[f]
    if order == 2:
        for f in factor_names:
            X[f"{f}^2"] = df[f] ** 2
        for f1, f2 in itertools.combinations(factor_names, 2):
            X[f"{f1}*{f2}"] = df[f1] * df[f2]
    X = sm.add_constant(X, has_constant="add")
    return X


def fit_model(df: pd.DataFrame, factor_names: list, response_name: str,
              order: int = 2) -> RSMModel:
    """
    Ajusta un modelo de 1er o 2do orden por minimos cuadrados (OLS).
    df debe contener las columnas de factores (codificados) y la respuesta.
    """
    import statsmodels.api as sm

    y = df[response_name].astype(float)
    Xb = _build_design_matrix(df, factor_names, order)
    model = sm.OLS(y, Xb, missing="drop").fit()

    return RSMModel(
        order=order,
        factor_names=factor_names,
        response_name=response_name,
        coefficients=model.params,
        fitted_model=model,
        X_design=Xb,
        y=y,
        r2=model.rsquared,
        r2_adj=model.rsquared_adj,
    )


# ---------------------------------------------------------------------------
# 4. ANOVA Y FALTA DE AJUSTE
# ---------------------------------------------------------------------------

def anova_table(rsm_model: RSMModel) -> pd.DataFrame:
    """
    Tabla ANOVA para el modelo ajustado: regresion, residual, total.
    """
    model = rsm_model.fitted_model
    y = rsm_model.y
    n = len(y)
    p = int(model.df_model) + 1  # numero de parametros (incluye intercepto)

    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_reg = float(model.ess)
    ss_res = float(model.ssr)
    df_reg = int(model.df_model)
    df_res = int(model.df_resid)
    df_total = n - 1

    ms_reg = ss_reg / df_reg if df_reg > 0 else np.nan
    ms_res = ss_res / df_res if df_res > 0 else np.nan
    f_val = ms_reg / ms_res if ms_res > 0 else np.nan
    p_val = 1 - stats.f.cdf(f_val, df_reg, df_res) if not np.isnan(f_val) else np.nan

    tabla = pd.DataFrame(
        {
            "Fuente": ["Regresion", "Residual", "Total"],
            "SC": [ss_reg, ss_res, ss_total],
            "GL": [df_reg, df_res, df_total],
            "CM": [ms_reg, ms_res, np.nan],
            "F": [f_val, np.nan, np.nan],
            "p-valor": [p_val, np.nan, np.nan],
        }
    )
    return tabla


def lack_of_fit_test(df: pd.DataFrame, rsm_model: RSMModel) -> dict:
    """
    Prueba de falta de ajuste usando las replicas de los puntos centrales
    (u otros puntos repetidos) como estimador de error puro.

    Retorna un diccionario con SC_LOF, SC_PE, GL, F y p-valor.
    Si no hay puntos repetidos, retorna None en los valores de F/p.
    """
    factor_names = rsm_model.factor_names
    response = rsm_model.response_name
    y = rsm_model.y.values
    y_hat = rsm_model.fitted_model.fittedvalues.values
    n = len(y)
    p = int(rsm_model.fitted_model.df_model) + 1

    # Identificar grupos de puntos repetidos (mismas coordenadas de factores)
    grp = df[factor_names].round(6).astype(str).agg("_".join, axis=1)
    groups = grp.groupby(grp).groups

    ss_pe = 0.0
    df_pe = 0
    for _, idx in groups.items():
        if len(idx) > 1:
            vals = df.loc[idx, response].astype(float).values
            ss_pe += float(np.sum((vals - vals.mean()) ** 2))
            df_pe += len(idx) - 1

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_lof = ss_res - ss_pe
    df_res = n - p
    df_lof = df_res - df_pe

    result = {
        "SC_residual": ss_res,
        "SC_falta_ajuste": ss_lof,
        "SC_error_puro": ss_pe,
        "GL_falta_ajuste": df_lof,
        "GL_error_puro": df_pe,
    }

    if df_pe > 0 and df_lof > 0:
        ms_lof = ss_lof / df_lof
        ms_pe = ss_pe / df_pe
        f_val = ms_lof / ms_pe if ms_pe > 0 else np.nan
        p_val = 1 - stats.f.cdf(f_val, df_lof, df_pe) if not np.isnan(f_val) else np.nan
        result["F"] = f_val
        result["p_valor"] = p_val
        result["conclusion"] = (
            "No hay evidencia de falta de ajuste (buen modelo)"
            if (not np.isnan(p_val) and p_val > 0.05)
            else "Posible falta de ajuste: considerar modelo de mayor orden"
        )
    else:
        result["F"] = None
        result["p_valor"] = None
        result["conclusion"] = "No hay suficientes puntos repetidos para estimar error puro"

    return result


def residual_diagnostics(rsm_model: RSMModel) -> pd.DataFrame:
    """Retorna dataframe con valores observados, predichos, residuos y residuos estandarizados."""
    model = rsm_model.fitted_model
    resid = model.resid
    std_resid = resid / np.std(resid, ddof=int(model.df_model) + 1)
    out = pd.DataFrame(
        {
            "Observado": rsm_model.y.values,
            "Predicho": model.fittedvalues.values,
            "Residuo": resid.values,
            "Residuo_estandarizado": std_resid.values,
        }
    )
    return out


# ---------------------------------------------------------------------------
# 5. ANALISIS CANONICO
# ---------------------------------------------------------------------------

def canonical_analysis(rsm_model: RSMModel) -> dict:
    """
    Analisis canonico de un modelo de 2do orden.
    Calcula el punto estacionario, los eigenvalores/eigenvectores de la
    matriz B, y clasifica el punto como maximo, minimo o silla (punto de montura).

    Solo valido para order=2.
    """
    if rsm_model.order != 2:
        raise ValueError("El analisis canonico requiere un modelo de 2do orden")

    factor_names = rsm_model.factor_names
    k = len(factor_names)
    coef = rsm_model.coefficients

    b0 = coef.get("const", 0.0)
    b = np.array([coef.get(f, 0.0) for f in factor_names])

    B = np.zeros((k, k))
    for i, f in enumerate(factor_names):
        B[i, i] = coef.get(f"{f}^2", 0.0)
    for (i, f1), (j, f2) in itertools.combinations(enumerate(factor_names), 2):
        val = coef.get(f"{f1}*{f2}", 0.0)
        B[i, j] = val / 2
        B[j, i] = val / 2

    # Punto estacionario: xs = -0.5 * B^-1 * b
    try:
        B_inv = np.linalg.inv(B)
        xs = -0.5 * B_inv @ b
    except np.linalg.LinAlgError:
        xs = np.full(k, np.nan)

    y_xs = b0 + 0.5 * b @ xs

    eigvals, eigvecs = np.linalg.eigh(B)

    if np.all(eigvals < -1e-9):
        tipo = "Maximo"
    elif np.all(eigvals > 1e-9):
        tipo = "Minimo"
    elif np.any(np.abs(eigvals) < 1e-9):
        tipo = "Cresta estacionaria (eigenvalor ~ 0)"
    else:
        tipo = "Punto de silla (montura)"

    return {
        "punto_estacionario": dict(zip(factor_names, xs)),
        "respuesta_en_punto_estacionario": y_xs,
        "eigenvalores": eigvals,
        "eigenvectores": eigvecs,
        "matriz_B": B,
        "tipo_punto": tipo,
    }


# ---------------------------------------------------------------------------
# 6. ASCENSO MAS PRONUNCIADO
# ---------------------------------------------------------------------------

def steepest_ascent(rsm_model: RSMModel, base_point: Optional[dict] = None,
                     step_factor: str = None, step_size: float = 1.0,
                     n_steps: int = 5, maximize: bool = True) -> pd.DataFrame:
    """
    Calcula la trayectoria de ascenso (o descenso) mas pronunciado a partir
    de un modelo de 1er orden (o los terminos lineales de un modelo de 2do orden).

    step_factor: nombre del factor usado como referencia de paso (por defecto,
                 el de mayor coeficiente absoluto).
    """
    factor_names = rsm_model.factor_names
    coef = rsm_model.coefficients
    b = np.array([coef.get(f, 0.0) for f in factor_names])
    if not maximize:
        b = -b

    if base_point is None:
        base_point = {f: 0.0 for f in factor_names}
    x0 = np.array([base_point[f] for f in factor_names])

    if step_factor is None:
        ref_idx = int(np.argmax(np.abs(b)))
    else:
        ref_idx = factor_names.index(step_factor)

    # Escalar el vector de direccion para que el paso en la variable de
    # referencia sea step_size
    direction = b / abs(b[ref_idx]) * step_size

    rows = []
    for step in range(0, n_steps + 1):
        x = x0 + step * direction
        pred_df = pd.DataFrame([dict(zip(factor_names, x))])
        y_pred = rsm_model.predict(pred_df)[0]
        row = {"Paso": step}
        row.update(dict(zip(factor_names, x)))
        row[f"{rsm_model.response_name}_predicho"] = y_pred
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. ANALISIS DE CRESTA (RIDGE ANALYSIS)
# ---------------------------------------------------------------------------

def ridge_analysis(rsm_model: RSMModel, radii: np.ndarray = None,
                    maximize: bool = True) -> pd.DataFrame:
    """
    Analisis de cresta: para cada radio (distancia al centro del diseno),
    encuentra el punto sobre la esfera de ese radio que optimiza la
    respuesta predicha, resolviendo (B - lambda*I) x = -0.5 b.
    """
    if rsm_model.order != 2:
        raise ValueError("El analisis de cresta requiere un modelo de 2do orden")

    factor_names = rsm_model.factor_names
    k = len(factor_names)
    coef = rsm_model.coefficients
    b0 = coef.get("const", 0.0)
    b = np.array([coef.get(f, 0.0) for f in factor_names])

    B = np.zeros((k, k))
    for i, f in enumerate(factor_names):
        B[i, i] = coef.get(f"{f}^2", 0.0)
    for (i, f1), (j, f2) in itertools.combinations(enumerate(factor_names), 2):
        val = coef.get(f"{f1}*{f2}", 0.0)
        B[i, j] = val / 2
        B[j, i] = val / 2

    if radii is None:
        radii = np.linspace(0.5, 2.0, 7)

    eigvals_all = np.linalg.eigvalsh(B)

    rows = []
    for r in radii:
        def eqs(lam):
            try:
                x = np.linalg.solve(B - lam * np.eye(k), -0.5 * b)
            except np.linalg.LinAlgError:
                return 1e6
            return np.sum(x ** 2) - r ** 2

        # Escanear un rango de lambda por fuera del espectro de B y localizar
        # un cambio de signo para acotar la raiz de forma robusta (metodo de
        # Lagrange para el analisis de cresta, Myers & Montgomery).
        if maximize:
            edge = eigvals_all.max()
            candidates = edge + np.geomspace(1e-3, 500, 400)
        else:
            edge = eigvals_all.min()
            candidates = edge - np.geomspace(1e-3, 500, 400)

        vals = np.array([eqs(c) for c in candidates])
        lam_sol = None
        sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
        if len(sign_changes) > 0:
            i0 = sign_changes[0]
            try:
                lam_sol = optimize.brentq(eqs, candidates[i0], candidates[i0 + 1], maxiter=200)
            except Exception:
                lam_sol = None
        if lam_sol is None:
            lam_sol = candidates[int(np.argmin(np.abs(vals)))]

        x_opt = np.linalg.solve(B - lam_sol * np.eye(k), -0.5 * b)
        y_opt = b0 + b @ x_opt + x_opt @ B @ x_opt

        row = {"Radio": r}
        row.update(dict(zip(factor_names, x_opt)))
        row[f"{rsm_model.response_name}_predicho"] = y_opt
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. DESEABILIDAD DE DERRINGER-SUICH (MULTI-RESPUESTA)
# ---------------------------------------------------------------------------

@dataclass
class ResponseGoal:
    name: str
    goal: str  # 'maximize', 'minimize', 'target'
    low: float
    high: float
    target: Optional[float] = None
    weight: float = 1.0       # importancia relativa en la media geometrica
    s_t: float = 1.0          # exponente de forma (s para max/min, t para target)


def individual_desirability(y: float, rg: ResponseGoal) -> float:
    """Calcula la deseabilidad individual d_i de Derringer-Suich para una respuesta."""
    lo, hi = rg.low, rg.high
    if rg.goal == "maximize":
        if y <= lo:
            return 0.0
        if y >= hi:
            return 1.0
        return ((y - lo) / (hi - lo)) ** rg.s_t
    elif rg.goal == "minimize":
        if y <= lo:
            return 1.0
        if y >= hi:
            return 0.0
        return ((hi - y) / (hi - lo)) ** rg.s_t
    elif rg.goal == "target":
        t = rg.target
        if y < lo or y > hi:
            return 0.0
        if y <= t:
            return ((y - lo) / (t - lo)) ** rg.s_t
        else:
            return ((hi - y) / (hi - t)) ** rg.s_t
    else:
        raise ValueError("goal debe ser 'maximize', 'minimize' o 'target'")


def overall_desirability(y_values: dict, goals: list) -> float:
    """
    Deseabilidad global D = (d1^w1 * d2^w2 * ... * dn^wn) ^ (1/sum(wi))
    (media geometrica ponderada de Derringer-Suich).
    """
    ds, ws = [], []
    for rg in goals:
        d = individual_desirability(y_values[rg.name], rg)
        ds.append(max(d, 1e-12))
        ws.append(rg.weight)
    ds = np.array(ds)
    ws = np.array(ws)
    if np.any(ds <= 1e-12):
        return 0.0
    D = np.prod(ds ** ws) ** (1 / np.sum(ws))
    return float(D)


def optimize_desirability(models: dict, goals: list, factor_names: list,
                           bounds: tuple = (-1.6, 1.6), n_starts: int = 30,
                           seed: int = 42) -> dict:
    """
    Optimiza numericamente la deseabilidad global sobre la region experimental,
    combinando las predicciones de varios modelos RSM (uno por respuesta).

    models: {'nombre_respuesta': RSMModel, ...}
    goals: lista de ResponseGoal
    bounds: limites en unidades codificadas para cada factor (mismo para todos)
    """
    rng = np.random.default_rng(seed)
    k = len(factor_names)

    def neg_desirability(x):
        pred_df = pd.DataFrame([dict(zip(factor_names, x))])
        y_vals = {}
        for rg in goals:
            model = models[rg.name]
            y_vals[rg.name] = model.predict(pred_df)[0]
        D = overall_desirability(y_vals, goals)
        return -D

    best = None
    for _ in range(n_starts):
        x0 = rng.uniform(bounds[0], bounds[1], size=k)
        res = optimize.minimize(
            neg_desirability, x0, method="L-BFGS-B",
            bounds=[bounds] * k,
        )
        if best is None or res.fun < best.fun:
            best = res

    x_opt = best.x
    pred_df = pd.DataFrame([dict(zip(factor_names, x_opt))])
    y_opt = {}
    for rg in goals:
        y_opt[rg.name] = models[rg.name].predict(pred_df)[0]

    return {
        "factores_optimos_codificados": dict(zip(factor_names, x_opt)),
        "respuestas_predichas": y_opt,
        "deseabilidad_global": -best.fun,
    }
