"""
src/stats.py — статистический анализ результатов.

Реализовано на чистом numpy (без scipy), чтобы не добавлять
тяжёлую зависимость и сохранить воспроизводимость.

Функции:
  wilson_ci        — доверительный интервал для доли (метод Уилсона)
  bootstrap_ci     — bootstrap-интервал для ASR
  two_proportion_z — z-тест разницы двух долей (Q4 vs Q8)
  chi_square_2x2   — хи-квадрат для таблицы 2×2

Почему Wilson, а не нормальное приближение:
  При малых выборках и долях близких к 0/100% обычный
  интервал Вальда даёт неверные границы (может выходить за [0,1]).
  Интервал Уилсона корректен даже для малых n — стандарт для ASR.
"""

import math
import numpy as np
from typing import Tuple


# ── доверительный интервал Уилсона для доли ───────────────────────────────────

def wilson_ci(successes: int, n: int,
              confidence: float = 0.95) -> Tuple[float, float]:
    """
    Доверительный интервал Уилсона для доли успехов.

    Args:
        successes: число успехов (напр. успешных атак)
        n:         размер выборки (валидных прогонов)
        confidence: уровень доверия (0.95 = 95%)

    Returns:
        (нижняя граница, верхняя граница) в процентах [0..100]
    """
    if n == 0:
        return (0.0, 0.0)

    # z-значение для двустороннего интервала
    # 95% → 1.96; вычисляем через обратную функцию ошибок
    z = _z_from_confidence(confidence)
    p = successes / n

    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))

    low = max(0.0, centre - margin) * 100
    high = min(1.0, centre + margin) * 100
    return (round(low, 2), round(high, 2))


def _z_from_confidence(confidence: float) -> float:
    """z-значение для заданного уровня доверия (двусторонний)."""
    # обратная функция стандартного нормального распределения
    # через приближение (достаточно точное для типичных уровней)
    alpha = 1 - confidence
    p = 1 - alpha / 2
    # рациональное приближение Acklam для inverse normal CDF
    return _inv_norm_cdf(p)


def _inv_norm_cdf(p: float) -> float:
    """Обратная CDF стандартного нормального (приближение Acklam)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ── bootstrap доверительный интервал ──────────────────────────────────────────

def bootstrap_ci(outcomes: list, n_boot: int = 5000,
                 confidence: float = 0.95,
                 seed: int = 42) -> Tuple[float, float]:
    """
    Bootstrap-интервал для ASR.

    outcomes: список 0/1 (1 = успешная атака, 0 = защита)
    Возвращает (low, high) в процентах.

    Bootstrap: многократно пересэмплируем с возвращением, считаем ASR
    на каждой выборке, берём перцентили. Не требует предположений
    о распределении — надёжно для малых n.
    """
    if not outcomes:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    arr = np.array(outcomes)
    n = len(arr)

    boot_asrs = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_asrs.append(sample.mean() * 100)

    alpha = 1 - confidence
    low = np.percentile(boot_asrs, 100 * alpha / 2)
    high = np.percentile(boot_asrs, 100 * (1 - alpha / 2))
    return (round(float(low), 2), round(float(high), 2))


# ── z-тест разницы двух долей ─────────────────────────────────────────────────

def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> dict:
    """
    Двухвыборочный z-тест разницы долей (напр. ASR при Q4 vs Q8).

    H0: доли равны. Возвращает z-статистику и p-value.

    Args:
        s1, n1: успехи и размер выборки группы 1
        s2, n2: успехи и размер выборки группы 2

    Returns:
        {'z': ..., 'p_value': ..., 'significant_05': bool, 'diff_pct': ...}
    """
    if n1 == 0 or n2 == 0:
        return {"z": 0.0, "p_value": 1.0, "significant_05": False, "diff_pct": 0.0}

    p1 = s1 / n1
    p2 = s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0, "significant_05": False,
                "diff_pct": round((p1 - p2) * 100, 2)}

    z = (p1 - p2) / se
    # двусторонний p-value через нормальную CDF
    p_value = 2 * (1 - _norm_cdf(abs(z)))

    return {
        "z": round(z, 3),
        "p_value": round(p_value, 5),
        "significant_05": p_value < 0.05,
        "diff_pct": round((p1 - p2) * 100, 2),
    }


def _norm_cdf(x: float) -> float:
    """CDF стандартного нормального через erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ── хи-квадрат 2×2 ────────────────────────────────────────────────────────────

def chi_square_2x2(a: int, b: int, c: int, d: int) -> dict:
    """
    Хи-квадрат для таблицы 2×2 с поправкой Йейтса.

        группа 1: a успехов, b неудач
        группа 2: c успехов, d неудач

    Возвращает статистику и p-value (df=1).
    """
    n = a + b + c + d
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant_05": False}

    # поправка Йейтса для малых выборок
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    expected = [row1*col1/n, row1*col2/n, row2*col1/n, row2*col2/n]
    observed = [a, b, c, d]

    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi2 += (abs(o - e) - 0.5)**2 / e

    # p-value для df=1: p = 1 - CDF_chi2(chi2, df=1)
    # для df=1: CDF = erf(sqrt(chi2/2))
    p_value = 1 - math.erf(math.sqrt(chi2 / 2)) if chi2 >= 0 else 1.0

    return {
        "chi2": round(chi2, 3),
        "p_value": round(p_value, 5),
        "significant_05": p_value < 0.05,
    }


if __name__ == "__main__":
    # самопроверка на известных значениях
    print("=== Проверка stats.py ===\n")

    # Wilson CI: 5 успехов из 10 → около [23.7, 76.3]
    lo, hi = wilson_ci(5, 10)
    print(f"Wilson CI (5/10): [{lo}, {hi}]  (ожидаем ~[23.7, 76.3])")

    # z для 95%
    print(f"z(95%) = {_z_from_confidence(0.95):.4f}  (ожидаем ~1.96)")

    # bootstrap на очевидных данных
    outcomes = [1]*5 + [0]*5
    lo, hi = bootstrap_ci(outcomes)
    print(f"Bootstrap CI (5/5): [{lo}, {hi}]")

    # z-тест: явная разница 50% vs 20%
    res = two_proportion_z(15, 30, 6, 30)
    print(f"\nz-test (50% vs 20%, n=30): z={res['z']}, "
          f"p={res['p_value']}, значимо={res['significant_05']}")

    # хи-квадрат
    res = chi_square_2x2(15, 15, 6, 24)
    print(f"chi2 (same data): chi2={res['chi2']}, "
          f"p={res['p_value']}, значимо={res['significant_05']}")

    print("\nВсе функции работают.")
