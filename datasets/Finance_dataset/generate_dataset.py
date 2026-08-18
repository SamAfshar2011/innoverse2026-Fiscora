#!/usr/bin/env python3
"""
generate_dataset.py — Synthetic-but-realistic personal-finance behaviour dataset
================================================================================
Builds `finance_behavior.csv`: a panel of USERS × MONTHS of coherent financial
behaviour, designed for the sequence model in ../../Financa.ipynb.

Why synthetic: there is no public, privacy-clean, per-user monthly behaviour panel
with income + 12-category spend + crypto + budgets. So we generate one with
*coherent* internal relationships (income ↔ spend ↔ savings ↔ budget adherence,
seasonal and weekend patterns, spending spikes, crypto holders), seeded for
reproducibility. Every user follows a persona; no row is internally inconsistent.

Prediction tasks the panel supports (see README.md):
  · regression  : next month's total spend
  · classification: will next month exceed budget? (overspend risk)

Run:  python generate_dataset.py            # writes finance_behavior.csv (+ prints summary)
"""
import os
import numpy as np
import pandas as pd

SEED = 20260707
rng = np.random.default_rng(SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "finance_behavior.csv")

N_USERS = 900
MONTHS = 30                      # months of history per user
CATEGORIES = ["food_dining", "groceries", "transport", "housing_utilities", "health",
              "entertainment", "shopping", "education", "subscriptions", "transfers",
              "income", "other"]
SPEND_CATS = [c for c in CATEGORIES if c != "income"]     # 11 spend categories

# Seasonal multiplier by calendar month (1..12): holidays & summer travel lift spend
SEASON = np.array([0.98, 0.95, 1.00, 1.02, 1.03, 1.06, 1.08, 1.05, 1.00, 1.02, 1.10, 1.20])
# Housing/utilities is a near-fixed monthly obligation; entertainment/shopping are elastic
ELASTICITY = {"food_dining": 0.9, "groceries": 0.5, "transport": 0.6, "housing_utilities": 0.1,
              "health": 0.7, "entertainment": 1.3, "shopping": 1.4, "education": 0.6,
              "subscriptions": 0.15, "transfers": 1.0, "other": 1.0}


def make_persona():
    """One coherent user archetype."""
    income_base = float(rng.uniform(2200, 9000))                 # monthly income (USD units)
    discipline = float(np.clip(rng.beta(2.2, 2.2), 0.05, 0.95))  # saving discipline 0..1
    volatility = float(np.clip(rng.beta(2, 5) * 0.6, 0.05, 0.5)) # month-to-month spend noise
    # base spend fraction of income shrinks with discipline
    spend_frac = float(np.clip(rng.normal(0.92 - 0.35 * discipline, 0.06), 0.45, 1.05))
    budget_ratio = float(np.clip(rng.normal(0.80 - 0.15 * discipline, 0.05), 0.45, 0.98))
    # category propensity (Dirichlet) — housing & groceries always meaningful
    alpha = np.array([rng.uniform(0.4, 2.5) for _ in SPEND_CATS])
    alpha[SPEND_CATS.index("housing_utilities")] += 3.0
    alpha[SPEND_CATS.index("groceries")] += 1.5
    propensity = rng.dirichlet(alpha)
    is_crypto = rng.random() < 0.45
    crypto0 = float(rng.uniform(300, 15000)) if is_crypto else 0.0
    weekend_bias = float(np.clip(rng.normal(0.34, 0.08), 0.15, 0.6))
    spike_prob = float(rng.uniform(0.03, 0.16))                  # chance of a spend spike month
    return dict(income_base=income_base, discipline=discipline, volatility=volatility,
                spend_frac=spend_frac, budget_ratio=budget_ratio, propensity=propensity,
                is_crypto=is_crypto, crypto=crypto0, weekend_bias=weekend_bias,
                spike_prob=spike_prob, start_month=int(rng.integers(0, 12)))


def gen_user(uid, p):
    rows = []
    crypto_val = p["crypto"]
    for m in range(MONTHS):
        cal_month = (p["start_month"] + m) % 12 + 1                # 1..12
        season = SEASON[cal_month - 1]
        income = p["income_base"] * float(rng.normal(1.0, 0.05))    # small income noise
        # occasional income bump (bonus) in month 6 & 12
        if cal_month in (6, 12) and rng.random() < 0.5:
            income *= float(rng.uniform(1.1, 1.6))

        spike = rng.random() < p["spike_prob"]
        spend_mult = season * float(rng.normal(1.0, p["volatility"])) * (1.0 + (0.6 if spike else 0.0))
        total_spend = max(50.0, p["income_base"] * p["spend_frac"] * spend_mult)

        # allocate across categories with per-category elastic noise
        weights = np.array([p["propensity"][i] *
                            (season ** ELASTICITY[c]) *
                            float(rng.normal(1.0, 0.18 * ELASTICITY[c]))
                            for i, c in enumerate(SPEND_CATS)])
        weights = np.clip(weights, 1e-3, None)
        weights /= weights.sum()
        cat_spend = {c: float(total_spend * w) for c, w in zip(SPEND_CATS, weights)}
        total_spend = float(sum(cat_spend.values()))              # reconcile

        budget = p["income_base"] * p["budget_ratio"]
        n_txn = int(np.clip(rng.poisson(total_spend / 55.0), 5, 400))
        weekend_ratio = float(np.clip(rng.normal(p["weekend_bias"], 0.05), 0.05, 0.75))
        largest_txn = float(total_spend * np.clip(rng.beta(2, 8) + (0.25 if spike else 0.0), 0.03, 0.6))
        savings_rate = float((income - total_spend) / income)

        # crypto holdings evolve as a volatile random walk (only for holders)
        if p["is_crypto"]:
            crypto_val = float(max(0.0, crypto_val * float(np.exp(rng.normal(0.01, 0.18)))))
            crypto_txn = int(rng.poisson(1.2))
        else:
            crypto_val, crypto_txn = 0.0, 0

        row = {"user_id": uid, "month_index": m, "month_of_year": cal_month,
               "income": round(income, 2), "spend": round(total_spend, 2),
               "budget": round(budget, 2),
               "budget_util": round(total_spend / budget, 4),
               "savings_rate": round(savings_rate, 4),
               "n_transactions": n_txn,
               "weekend_ratio": round(weekend_ratio, 4),
               "largest_txn": round(largest_txn, 2),
               "crypto_value": round(crypto_val, 2),
               "crypto_txn": crypto_txn,
               "spike": int(spike),
               "overspend": int(total_spend > budget)}
        for c in SPEND_CATS:
            row[f"cat_{c}"] = round(cat_spend[c], 2)
        rows.append(row)
    return rows


def main():
    all_rows = []
    for uid in range(N_USERS):
        all_rows.extend(gen_user(uid, make_persona()))
    df = pd.DataFrame(all_rows)
    # column order: keys first, then category columns
    lead = ["user_id", "month_index", "month_of_year", "income", "spend", "budget",
            "budget_util", "savings_rate", "n_transactions", "weekend_ratio",
            "largest_txn", "crypto_value", "crypto_txn", "spike", "overspend"]
    df = df[lead + [f"cat_{c}" for c in SPEND_CATS]]
    df.to_csv(CSV_PATH, index=False)

    print(f"wrote {CSV_PATH}")
    print(f"  users={df.user_id.nunique()}  months/user={MONTHS}  rows={len(df)}  cols={df.shape[1]}")
    print(f"  overspend rate = {df.overspend.mean():.3f}  |  mean savings_rate = {df.savings_rate.mean():.3f}")
    print(f"  mean monthly income = {df.income.mean():,.0f}  spend = {df.spend.mean():,.0f}")
    print(f"  crypto holders (rows w/ value>0) = {(df.crypto_value > 0).mean():.2%}")


if __name__ == "__main__":
    main()
