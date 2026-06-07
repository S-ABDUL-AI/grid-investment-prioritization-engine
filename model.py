"""
Grid Investment Prioritization Engine — data generation and portfolio optimization.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── DESIGN TOKENS ─────────────────────────────────────────────────────────────
NAVY      = "#0A1F44"
NAVY_MID  = "#152B5C"
GOLD      = "#C9A84C"
GOLD_LT   = "#E8C97A"
INK       = "#1A1A1A"
BODY      = "#2C3E50"
MUTED     = "#6B7280"
RED       = "#C8382A"
AMBER     = "#B8560A"
GREEN     = "#1A7A2E"
TEAL      = "#0E7490"
RULE      = "#E2E6EC"
OFF_WHITE = "#F8F6F1"

REGIONS = [
    "LA Foothills",
    "San Bernardino Mtns",
    "Inland Empire",
    "San Gabriel Valley",
    "Orange County North",
    "Coachella Valley",
    "Pomona–Ontario",
]

PROJECT_CATEGORY_CONFIG = {
    "Wildfire Mitigation":    {"n": 22, "cost_range": (0.8e6,  18e6),  "base_bcr": 3.8},
    "Grid Hardening":         {"n": 18, "cost_range": (1.5e6,  25e6),  "base_bcr": 2.9},
    "Predictive Maintenance": {"n": 16, "cost_range": (0.3e6,  6e6),   "base_bcr": 4.2},
    "Undergrounding":         {"n": 12, "cost_range": (5e6,    60e6),  "base_bcr": 1.8},
    "Substation Upgrades":    {"n": 10, "cost_range": (3e6,    40e6),  "base_bcr": 2.4},
    "Vegetation Management":  {"n": 14, "cost_range": (0.5e6,  8e6),   "base_bcr": 5.1},
    "Climate Resilience":     {"n":  8, "cost_range": (2e6,    20e6),  "base_bcr": 2.2},
}

CATEGORIES = sorted(PROJECT_CATEGORY_CONFIG.keys())

CAT_COLORS = {
    "Wildfire Mitigation":    RED,
    "Grid Hardening":         NAVY,
    "Predictive Maintenance": GREEN,
    "Undergrounding":         "#6a1b9a",
    "Substation Upgrades":    TEAL,
    "Vegetation Management":  "#558b2f",
    "Climate Resilience":     "#00838f",
}

OBJ_MAP = {
    "Maximize Risk Reduction":      "risk_per_dollar",
    "Maximize BCR":                 "bcr",
    "Maximize Customers Protected": "customers_protected",
}


def _project_name(cat, pid, rng):
    names = {
        "Wildfire Mitigation":    ["Covered Conductor Replacement", "PSPS Automation Upgrade",
                                   "Fire Hardening Program", "Ignition Prevention Initiative",
                                   "High Fire Risk Zone Retrofit"],
        "Grid Hardening":         ["Transmission Reinforcement", "Distribution Upgrade",
                                   "Storm Hardening Program", "Infrastructure Modernization",
                                   "Grid Resilience Initiative"],
        "Predictive Maintenance": ["Sensor Deployment Program", "AI-Driven Inspection Rollout",
                                   "Condition Monitoring Upgrade", "Smart Asset Diagnostics",
                                   "Remote Monitoring Initiative"],
        "Undergrounding":         ["Underground Conversion", "Overhead-to-Underground Migration",
                                   "Urban Circuit Undergrounding", "Residential Underground Program"],
        "Substation Upgrades":    ["Substation Automation", "Transformer Replacement Program",
                                   "Protection System Upgrade", "Control System Modernization"],
        "Vegetation Management":  ["Enhanced Patrol Program", "Rapid Response Vegetation",
                                   "LiDAR Vegetation Mapping", "Fuel Load Reduction",
                                   "Clearance Enhancement Program"],
        "Climate Resilience":     ["Extreme Heat Adaptation", "Drought Resilience Program",
                                   "Climate-Ready Infrastructure", "Long-Term Resilience Plan"],
    }
    return f"{rng.choice(names[cat])} {pid}"


def generate_projects(seed=42):
    rng = np.random.default_rng(seed)

    rows = []
    proj_id = 1
    for cat, cfg in PROJECT_CATEGORY_CONFIG.items():
        for _ in range(cfg["n"]):
            cost = rng.uniform(*cfg["cost_range"])
            bcr_noise = rng.normal(0, 0.6)
            bcr = max(0.5, cfg["base_bcr"] + bcr_noise)

            total_benefit = bcr * cost
            direct_frac = rng.uniform(0.55, 0.82)
            risk_reduction = total_benefit * direct_frac
            cobenefit      = total_benefit * (1 - direct_frac)

            pf_reduction = np.clip(rng.normal(0.18, 0.07), 0.04, 0.40)

            duration_months = int(rng.choice([6, 12, 18, 24, 36],
                                  p=[0.15, 0.30, 0.25, 0.20, 0.10]))

            reg_required = rng.random() < (
                0.7 if cat in ["Wildfire Mitigation", "Vegetation Management"] else 0.25
            )

            assets_protected = int(rng.integers(5, 280))
            customers = int(rng.integers(500, 85_000))

            annual_benefit = risk_reduction / (duration_months / 12)
            years = duration_months / 12
            npv = sum(annual_benefit / (1.07 ** t) for t in range(1, int(years) + 2)) - cost

            co2_reduction = int(rng.integers(10, 2500))

            rows.append({
                "project_id":          f"INV-{proj_id:03d}",
                "project_name":        _project_name(cat, proj_id, rng),
                "category":            cat,
                "region":              rng.choice(REGIONS),
                "cost_usd":            round(cost, 0),
                "risk_reduction_usd":  round(risk_reduction, 0),
                "cobenefit_usd":       round(cobenefit, 0),
                "total_benefit_usd":   round(total_benefit, 0),
                "bcr":                 round(bcr, 2),
                "npv_usd":             round(npv, 0),
                "pf_reduction":        round(pf_reduction, 3),
                "duration_months":     duration_months,
                "assets_protected":    assets_protected,
                "customers_protected": customers,
                "regulatory_required": reg_required,
                "co2_reduction_tonnes":co2_reduction,
                "cost_per_customer":   round(cost / customers, 2),
                "risk_per_dollar":     round(risk_reduction / cost, 3),
            })
            proj_id += 1

    return pd.DataFrame(rows)


def greedy_optimize(df_cand, budget, obj_col, budget_col="cost_usd"):
    df_sorted = df_cand.sort_values(obj_col, ascending=False).copy()
    selected, spent = [], 0
    for _, row in df_sorted.iterrows():
        if spent + row[budget_col] <= budget:
            selected.append(row["project_id"])
            spent += row[budget_col]
    return selected


def run_monte_carlo(projects_df, selected_ids, n_sims=2000, seed=99):
    rng = np.random.default_rng(seed)
    sel = projects_df[projects_df["project_id"].isin(selected_ids)]
    if sel.empty:
        return None
    results = []
    for _ in range(n_sims):
        noise      = rng.normal(1.0, 0.15, len(sel))
        sim_bcr    = (sel["bcr"].values * noise).mean()
        noise2     = rng.normal(1.0, 0.18, len(sel))
        sim_risk   = (sel["risk_reduction_usd"].values * noise2).sum()
        results.append({"sim_bcr": sim_bcr, "sim_risk_reduction": sim_risk})
    return pd.DataFrame(results)


def build_scenarios(dff, scenario_budget, prioritize_by, include_reg):
    obj_col = OBJ_MAP[prioritize_by]

    scen_nothing = {
        "name": "Do Nothing", "ids": [], "cost": 0,
        "risk_red": 0, "customers": 0, "bcr": 0,
    }

    if include_reg:
        reg_projects = dff[dff["regulatory_required"]].copy()
        reg_cost = reg_projects["cost_usd"].sum()
        remaining = scenario_budget - reg_cost
        opt_cand = dff[~dff["regulatory_required"]].copy()
        opt_ids = reg_projects["project_id"].tolist()
        if remaining > 0:
            opt_ids += greedy_optimize(opt_cand, remaining, obj_col)
    else:
        opt_ids = greedy_optimize(dff, scenario_budget, obj_col)

    opt_sel = dff[dff["project_id"].isin(opt_ids)]
    scen_opt = {
        "name": "Optimized Portfolio",
        "ids": opt_ids,
        "cost": opt_sel["cost_usd"].sum(),
        "risk_red": opt_sel["risk_reduction_usd"].sum(),
        "customers": opt_sel["customers_protected"].sum(),
        "bcr": (
            opt_sel["total_benefit_usd"].sum() / opt_sel["cost_usd"].sum()
            if opt_sel["cost_usd"].sum() > 0 else 0
        ),
    }

    reg_sel = dff[dff["regulatory_required"]]
    scen_reg = {
        "name": "Regulatory Minimum",
        "ids": reg_sel["project_id"].tolist(),
        "cost": reg_sel["cost_usd"].sum(),
        "risk_red": reg_sel["risk_reduction_usd"].sum(),
        "customers": reg_sel["customers_protected"].sum(),
        "bcr": (
            reg_sel["total_benefit_usd"].sum() / reg_sel["cost_usd"].sum()
            if reg_sel["cost_usd"].sum() > 0 else 0
        ),
    }

    return [scen_nothing, scen_reg, scen_opt], opt_sel, opt_ids, scen_opt, scen_reg


def scenario_summary_rows(scenarios, dff):
    rows = []
    baseline_risk = dff["risk_reduction_usd"].sum()
    for s in scenarios:
        roi = (s["risk_red"] / s["cost"] * 100) if s["cost"] > 0 else 0
        pct = (s["risk_red"] / baseline_risk * 100) if baseline_risk > 0 else 0
        rows.append({
            "Scenario": s["name"],
            "Projects Funded": len(s["ids"]),
            "Total Investment": f"${s['cost']/1e6:.1f}M",
            "Risk Reduction": f"${s['risk_red']/1e6:.1f}M",
            "% of Max Risk Red.": f"{pct:.1f}%",
            "Portfolio BCR": f"{s['bcr']:.2f}x",
            "ROI (Risk/Cost)": f"{roi:.0f}%",
            "Customers Protected": f"{s['customers']:,}",
        })
    return rows
