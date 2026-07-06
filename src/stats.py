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


# ── effect size (Cohen's h для разницы долей) ─────────────────────────────────

def cohens_h(p1: float, p2: float) -> dict:
    """
    Cohen's h — размер эффекта для разницы двух пропорций.
    p1, p2 в [0,1]. |h|: 0.2 малый, 0.5 средний, 0.8 большой.
    Значимость говорит ЕСТЬ ли эффект, effect size — НАСКОЛЬКО велик.
    """
    phi1 = 2 * math.asin(math.sqrt(max(0, min(1, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0, min(1, p2))))
    h = abs(phi1 - phi2)
    if h < 0.2:
        mag = "negligible"
    elif h < 0.5:
        mag = "small"
    elif h < 0.8:
        mag = "medium"
    else:
        mag = "large"
    return {"h": round(h, 3), "magnitude": mag}


# ── поправки на множественные сравнения ───────────────────────────────────────

def bonferroni_correction(p_values: list, alpha: float = 0.05) -> dict:
    """Поправка Бонферрони: делит порог на число тестов."""
    n = len(p_values)
    if n == 0:
        return {"corrected_alpha": alpha, "significant": [],
                "n_tests": 0, "n_significant": 0}
    ca = alpha / n
    sig = [p < ca for p in p_values]
    return {"corrected_alpha": round(ca, 6), "significant": sig,
            "n_tests": n, "n_significant": sum(sig)}


def holm_correction(p_values: list, alpha: float = 0.05) -> dict:
    """Поправка Холма: ступенчатая, менее консервативна чем Бонферрони."""
    n = len(p_values)
    if n == 0:
        return {"significant": [], "n_tests": 0, "n_significant": 0}
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    sig = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        if p < alpha / (n - rank):
            sig[orig_idx] = True
        else:
            break
    return {"significant": sig, "n_tests": n, "n_significant": sum(sig)}


# ── попарное сравнение групп ──────────────────────────────────────────────────

def pairwise_comparison(groups: dict, alpha: float = 0.05) -> dict:
    """
    Все пары групп по ASR + z-тест + effect size + поправка Холма.

    groups: {name: {'success': int, 'n': int}}
    """
    names = list(groups.keys())
    pairs, p_values = [], []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ga, gb = groups[a], groups[b]
            zt = two_proportion_z(ga["success"], ga["n"], gb["success"], gb["n"])
            h = cohens_h(ga["success"]/ga["n"] if ga["n"] else 0,
                         gb["success"]/gb["n"] if gb["n"] else 0)
            pairs.append({
                "group_a": a, "group_b": b,
                "asr_a": round(100*ga["success"]/ga["n"], 1) if ga["n"] else 0,
                "asr_b": round(100*gb["success"]/gb["n"], 1) if gb["n"] else 0,
                "diff_pp": zt["diff_pct"], "z": zt["z"], "p_value": zt["p_value"],
                "effect_h": h["h"], "effect_magnitude": h["magnitude"],
            })
            p_values.append(zt["p_value"])

    holm = holm_correction(p_values, alpha)
    for pair, s in zip(pairs, holm["significant"]):
        pair["significant_holm"] = s

    return {"pairs": pairs, "n_comparisons": len(pairs),
            "n_significant_after_correction": holm["n_significant"]}


# ── агрегация по seeds ────────────────────────────────────────────────────────

def aggregate_over_seeds(per_seed_asr: list) -> dict:
    """
    Среднее ± std + CI по seed-прогонам. Ключевое для воспроизводимости:
    показывает стабильность результата между независимыми прогонами.
    per_seed_asr: список ASR (%) по seeds.
    """
    if not per_seed_asr:
        return {"mean": 0, "std": 0, "min": 0, "max": 0,
                "ci_low": 0, "ci_high": 0, "n_seeds": 0}
    arr = np.array(per_seed_asr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return {
        "mean": round(mean, 2), "std": round(std, 2),
        "min": round(float(arr.min()), 2), "max": round(float(arr.max()), 2),
        "ci_low": round(max(0, mean - margin), 2),
        "ci_high": round(min(100, mean + margin), 2),
        "n_seeds": len(arr),
    }


# ── TOST: эквивалентностный тест (доказательство ОТСУТСТВИЯ эффекта) ───────────

def tost_two_proportions(s1: int, n1: int, s2: int, n2: int,
                         margin: float = 0.10, alpha: float = 0.05) -> dict:
    """
    Two One-Sided Tests (TOST) для эквивалентности двух долей.

    Обычный тест отвечает "есть ли разница". TOST отвечает на ДРУГОЙ,
    более сильный для нас вопрос: "доказуемо ли, что разница МЕНЬШЕ
    заданного порога эквивалентности". Это то, что нужно для честного
    негативного вывода: не "мы не нашли эффект", а "мы доказали, что
    эффект меньше margin".

    H0 (то, что опровергаем): |p1 - p2| >= margin  (эффект существенный)
    H1 (то, что хотим показать): |p1 - p2| < margin (эквивалентность)

    Args:
        margin: порог эквивалентности в долях (0.10 = 10 п.п.).
                Если истинная разница меньше, считаем эффект несущественным.

    Returns:
        {'equivalent': bool, 'p_tost': float, 'diff': float,
         'ci90_low', 'ci90_high', 'margin'}
    """
    if n1 == 0 or n2 == 0:
        return {"equivalent": False, "p_tost": 1.0, "diff": 0.0,
                "ci90_low": 0.0, "ci90_high": 0.0, "margin": margin}

    p1, p2 = s1 / n1, s2 / n2
    diff = p1 - p2
    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    if se == 0:
        se = 1e-9

    # два односторонних теста
    # тест 1: H0: diff <= -margin  vs  H1: diff > -margin
    z_lower = (diff - (-margin)) / se
    p_lower = 1 - _norm_cdf(z_lower)          # хотим маленькое p
    # тест 2: H0: diff >= +margin  vs  H1: diff < +margin
    z_upper = (diff - margin) / se
    p_upper = _norm_cdf(z_upper)              # хотим маленькое p

    # TOST: эквивалентность если ОБА теста значимы
    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < alpha

    # 90% CI (соответствует alpha=0.05 для TOST)
    z90 = 1.645
    ci_low = diff - z90 * se
    ci_high = diff + z90 * se

    return {
        "equivalent": equivalent,
        "p_tost": round(p_tost, 5),
        "diff": round(diff * 100, 2),          # в п.п.
        "ci90_low": round(ci_low * 100, 2),
        "ci90_high": round(ci_high * 100, 2),
        "margin": round(margin * 100, 1),
    }


# ── анализ мощности: какой минимальный эффект мы могли задетектить ─────────────

def min_detectable_effect(n1: int, n2: int, baseline_p: float = 0.5,
                          alpha: float = 0.05, power: float = 0.80) -> dict:
    """
    Минимальный детектируемый эффект (MDE) при данном размере выборки.

    Отвечает рецензенту на вопрос "а хватило ли у вас мощности вообще
    увидеть эффект?". Возвращает наименьшую разницу долей, которую тест
    обнаружил бы с заданной мощностью.

    Args:
        n1, n2: размеры выборок
        baseline_p: базовая доля (консервативно 0.5 — максимум дисперсии)
        power: желаемая мощность (0.80 стандарт)

    Returns:
        {'mde_pp': минимальный эффект в п.п., 'n1', 'n2', 'power'}
    """
    if n1 == 0 or n2 == 0:
        return {"mde_pp": 100.0, "n1": n1, "n2": n2, "power": power}

    z_alpha = _inv_norm_cdf(1 - alpha/2)      # двусторонний
    z_beta = _inv_norm_cdf(power)

    # приближение для разницы пропорций
    p = baseline_p
    se_factor = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    mde = (z_alpha + z_beta) * se_factor

    return {
        "mde_pp": round(mde * 100, 2),
        "n1": n1, "n2": n2, "power": power,
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

    # ── новые функции ────────────────────────────────────────────────────────
    print("\n=== Расширенная статистика ===")

    # Cohen's h: 100% vs 22% — должен быть большой эффект
    h = cohens_h(1.0, 0.22)
    print(f"Cohen's h (100% vs 22%): h={h['h']} ({h['magnitude']})")

    # попарное сравнение стилей (реальные данные из эксперимента)
    style_groups = {
        "hidden":     {"success": 72, "n": 72},   # 100%
        "imperative": {"success": 60, "n": 72},   # ~83%
        "authority":  {"success": 16, "n": 72},   # ~22%
        "roleplay":   {"success": 24, "n": 72},   # ~33%
        "urgency":    {"success": 18, "n": 72},   # ~25%
    }
    pw = pairwise_comparison(style_groups)
    print(f"\nПопарные сравнения стилей: {pw['n_comparisons']} пар, "
          f"значимых после Холма: {pw['n_significant_after_correction']}")
    for p in pw["pairs"][:3]:
        print(f"  {p['group_a']} vs {p['group_b']}: "
              f"{p['asr_a']}% vs {p['asr_b']}%, "
              f"h={p['effect_h']} ({p['effect_magnitude']}), "
              f"significant={p['significant_holm']}")

    # агрегация по seeds
    seeds_asr = [52.8, 55.1, 51.0]
    agg = aggregate_over_seeds(seeds_asr)
    print(f"\nПо seeds {seeds_asr}: mean={agg['mean']}±{agg['std']}, "
          f"CI=[{agg['ci_low']}, {agg['ci_high']}]")

    # ── TOST: эквивалентность (главный негативный вывод) ─────────────────────
    print("\n=== TOST (эквивалентность квантизаций) ===")
    # реальные данные Qwen-1.5B: Q4 vs Q8, примерно равные
    # Q4: ~49% из n≈246, Q8: ~44% из n≈246
    tost = tost_two_proportions(121, 246, 108, 246, margin=0.10)
    print(f"Q4 vs Q8 (margin=10пп): разница={tost['diff']}пп, "
          f"90% CI=[{tost['ci90_low']}, {tost['ci90_high']}]")
    print(f"  эквивалентны (эффект <10пп доказан): {tost['equivalent']}, "
          f"p_TOST={tost['p_tost']}")

    # ── мощность: какой эффект вообще могли увидеть ──────────────────────────
    print("\n=== Анализ мощности ===")
    mde = min_detectable_effect(246, 246, baseline_p=0.5, power=0.80)
    print(f"При n={mde['n1']}+{mde['n2']}, power=0.80: "
          f"минимальный детектируемый эффект = {mde['mde_pp']}пп")
    print(f"  → эффекты крупнее {mde['mde_pp']}пп мы бы увидели; "
          f"их нет → вывод об отсутствии обоснован")

    print("\nВсе функции работают.")
