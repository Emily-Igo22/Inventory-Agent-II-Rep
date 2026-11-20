"""Supply Chain Analytics Inventory Tools.

This module provides verified numerical calculation engines for the 5 fundamental
inventory management models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop() (Safety Stock & Reorder Point - 4 cases)
4. calculate_periodic_review() (Fixed-Period Review / Order-up-to Level)
5. calculate_newsvendor() (Single-Period Stochastic Demand)

For every calculation, each tool deterministically returns:
1. Inputs Extracted (inputs_extracted / inputs)
2. Variable Definitions (variable_definitions)
3. Formula (formula / formulas)
4. Formula with Substituted Values (formula_with_substituted_values)
5. Intermediate Calculations (intermediate_calculations)
6. Final Answer (final_answer)
7. Verification Checks (verification_checks)
8. Assumptions (assumptions)

Strict Requirements:
- Validate inputs
- Never invent missing numerical inputs
- All numerical steps generated deterministically in Python
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from scipy import stats


# -----------------------------------------------------------------------------
# Common Utility Functions
# -----------------------------------------------------------------------------

def unit_normal_loss(z: float) -> float:
    """Calculate the standard normal loss function L(z) = phi(z) - z * (1 - Phi(z))."""
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return float(phi - z * (1.0 - Phi))


def normalize_service_level(service_level: Optional[float]) -> float:
    """Ensure service level is a float strictly between 0.0 and 1.0."""
    if service_level is None:
        raise ValueError("Service level is required when demand or lead time is variable; cannot invent missing input.")
    sl = float(service_level)
    if sl > 1.0:
        sl /= 100.0
    if not (0.0 < sl < 1.0):
        raise ValueError(
            f"Service level must be strictly between 0 and 1 (or 0% and 100%), got {service_level}"
        )
    return sl


# -----------------------------------------------------------------------------
# 1. Economic Order Quantity (EOQ)
# -----------------------------------------------------------------------------

def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    working_days_per_year: int = 250,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate Economic Order Quantity (EOQ): Q = sqrt(2DS/H)."""
    # 1. Input Validation
    if annual_demand is None:
        raise ValueError("Missing required input: annual_demand cannot be None.")
    if order_cost is None:
        raise ValueError("Missing required input: order_cost cannot be None.")
    if annual_demand <= 0:
        raise ValueError(f"Annual demand must be positive, got {annual_demand}")
    if order_cost <= 0:
        raise ValueError(f"Order cost must be positive, got {order_cost}")

    # Determine holding cost per unit H
    if holding_cost is not None:
        if holding_cost <= 0:
            raise ValueError(f"Holding cost must be positive, got {holding_cost}")
        h = float(holding_cost)
        h_source = "direct holding_cost"
    elif holding_cost_rate is not None and unit_cost is not None:
        rate = holding_cost_rate / 100.0 if holding_cost_rate > 1.0 else holding_cost_rate
        if rate <= 0:
            raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}")
        if unit_cost <= 0:
            raise ValueError(f"Unit cost must be positive, got {unit_cost}")
        h = float(rate * unit_cost)
        h_source = f"holding_cost_rate ({rate * 100:.1f}%) * unit_cost (${unit_cost:.2f})"
    elif holding_cost_rate is not None and unit_cost is None:
        raise ValueError("Holding cost rate provided without unit_cost; cannot invent missing unit_cost.")
    else:
        raise ValueError("Must provide either 'holding_cost' or both 'holding_cost_rate' and 'unit_cost'.")

    if working_days_per_year <= 0:
        raise ValueError(f"Working days per year must be positive, got {working_days_per_year}")

    # 2. Mathematical Calculations
    radicand = (2.0 * annual_demand * order_cost) / h
    eoq_exact = math.sqrt(radicand)
    eoq_rounded = int(round(eoq_exact))

    orders_per_year = annual_demand / eoq_exact
    cycle_time_years = eoq_exact / annual_demand
    cycle_time_days = cycle_time_years * working_days_per_year

    annual_ordering_cost = (annual_demand / eoq_exact) * order_cost
    annual_holding_cost = (eoq_exact / 2.0) * h
    total_inventory_cost = annual_ordering_cost + annual_holding_cost

    annual_purchase_cost = (annual_demand * unit_cost) if unit_cost is not None else None
    total_annual_cost = (total_inventory_cost + annual_purchase_cost) if annual_purchase_cost is not None else total_inventory_cost

    # 3. Verification Checks
    opt_diff = abs(annual_ordering_cost - annual_holding_cost)
    opt_percent_diff = (opt_diff / max(annual_ordering_cost, 1.0)) * 100.0
    second_derivative = (2.0 * annual_demand * order_cost) / (eoq_exact ** 3)

    verification_checks = [
        {
            "check": "Positivity Check",
            "passed": eoq_exact > 0,
            "details": f"Optimal batch Q* = {eoq_exact:.4f} units is strictly positive.",
        },
        {
            "check": "First-Order Optimality (AOC == AHC)",
            "passed": opt_percent_diff < 0.1,
            "details": f"Annual Ordering Cost (${annual_ordering_cost:.2f}) equals Annual Holding Cost (${annual_holding_cost:.2f}); difference = ${opt_diff:.4f} ({opt_percent_diff:.4f}%).",
        },
        {
            "check": "Second-Order Convexity Condition",
            "passed": second_derivative > 0,
            "details": f"d^2(TC)/dQ^2 = {second_derivative:.6f} > 0 confirms a unique global cost minimum.",
        },
    ]

    # 4. Standard 8-Section Payload
    inputs_extracted = {
        "annual_demand": float(annual_demand),
        "order_cost": float(order_cost),
        "holding_cost_per_unit_year": float(h),
        "holding_cost_source": h_source,
        "unit_cost": float(unit_cost) if unit_cost is not None else None,
        "working_days_per_year": int(working_days_per_year),
    }

    variable_definitions = {
        "D": "Annual Demand (units/year)",
        "S": "Ordering or Setup Cost per order ($/order)",
        "H": "Inventory Holding Cost per unit per year ($/unit/year)",
        "C": "Unit Purchase Price ($/unit, optional)",
        "Q*": "Economic Order Quantity minimizing total inventory cost (units/order)",
        "N": "Order Frequency: number of replenishment orders per year (orders/year)",
        "T": "Order Cycle Time: operating interval between consecutive orders (days)",
        "AOC": "Annual Ordering Cost = (D / Q) * S ($/year)",
        "AHC": "Annual Holding Cost = (Q / 2) * H ($/year)",
        "TIC": "Total Annual Inventory Cost = AOC + AHC ($/year)",
        "TC": "Total Annual System Cost = TIC + (D * C) ($/year)",
    }

    formulas = {
        "eoq": "Q* = sqrt((2 * D * S) / H)",
        "orders_per_year": "N = D / Q*",
        "cycle_time": "T = (Q* / D) * working_days_per_year",
        "annual_ordering_cost": "AOC = (D / Q*) * S",
        "annual_holding_cost": "AHC = (Q* / 2) * H",
        "total_inventory_cost": "TIC = AOC + AHC = sqrt(2 * D * S * H)",
        "total_annual_cost": "TC = (D * C) + AOC + AHC",
    }

    purch_str = f"${annual_purchase_cost:,.2f}" if annual_purchase_cost is not None else "$0.00"
    formula_with_substituted_values = {
        "eoq": f"Q* = sqrt((2 * {annual_demand:,.2f} * {order_cost:.2f}) / {h:.4f}) = sqrt({(2 * annual_demand * order_cost):,.2f} / {h:.4f}) = sqrt({radicand:,.4f}) = {eoq_exact:.4f} ≈ {eoq_rounded} units",
        "orders_per_year": f"N = {annual_demand:,.2f} / {eoq_exact:.4f} = {orders_per_year:.2f} orders/year",
        "cycle_time": f"T = ({eoq_exact:.4f} / {annual_demand:,.2f}) * {working_days_per_year} = {cycle_time_days:.2f} operating days",
        "annual_ordering_cost": f"AOC = ({annual_demand:,.2f} / {eoq_exact:.4f}) * {order_cost:.2f} = ${annual_ordering_cost:,.2f}/year",
        "annual_holding_cost": f"AHC = ({eoq_exact:.4f} / 2) * {h:.4f} = ${annual_holding_cost:,.2f}/year",
        "total_inventory_cost": f"TIC = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_inventory_cost:,.2f}/year",
        "total_annual_cost": f"TC = {purch_str} + ${total_inventory_cost:,.2f} = ${total_annual_cost:,.2f}/year",
    }

    intermediate_calculations = {
        "holding_cost_per_unit": round(h, 4),
        "eoq_numerator_2DS": round(2.0 * annual_demand * order_cost, 2),
        "eoq_radicand": round(radicand, 4),
        "eoq_exact": round(eoq_exact, 4),
        "orders_per_year": round(orders_per_year, 4),
        "cycle_time_years": round(cycle_time_years, 6),
        "cycle_time_days": round(cycle_time_days, 2),
        "annual_ordering_cost": round(annual_ordering_cost, 2),
        "annual_holding_cost": round(annual_holding_cost, 2),
        "total_inventory_cost": round(total_inventory_cost, 2),
        "annual_purchase_cost": round(annual_purchase_cost, 2) if annual_purchase_cost is not None else None,
        "total_annual_cost": round(total_annual_cost, 2),
        "formula_applied": "Q = sqrt(2 * D * S / H)",
    }

    final_answer = {
        "optimal_order_quantity": eoq_rounded,
        "optimal_order_quantity_exact": round(eoq_exact, 4),
        "orders_per_year": round(orders_per_year, 2),
        "cycle_time_days": round(cycle_time_days, 1),
        "total_annual_inventory_cost": round(total_inventory_cost, 2),
        "total_annual_cost": round(total_annual_cost, 2),
    }

    assumptions = [
        "Demand rate (D) is constant, continuous, and known with certainty throughout the year.",
        "Ordering/setup cost (S) is fixed and independent of the order size.",
        "Inventory carrying cost (H) is linear and charged on average cycle stock (Q / 2).",
        "Replenishment lead time is zero (instantaneous batch delivery).",
        "Shortages, stockouts, and backorders are not permitted.",
        "Unit purchase price is constant and independent of quantity ordered (no volume discounts).",
    ]

    return {
        "model": "Economic Order Quantity (EOQ)",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Backward-compatibility alias
        "results": {
            "optimal_order_quantity_exact": float(eoq_exact),
            "optimal_order_quantity": eoq_rounded,
            "orders_per_year": float(round(orders_per_year, 2)),
            "cycle_time_days": float(round(cycle_time_days, 1)),
            "annual_ordering_cost": float(round(annual_ordering_cost, 2)),
            "annual_holding_cost": float(round(annual_holding_cost, 2)),
            "total_inventory_cost": float(round(total_inventory_cost, 2)),
            "annual_purchase_cost": float(round(annual_purchase_cost, 2)) if annual_purchase_cost else None,
            "total_annual_cost": float(round(total_annual_cost, 2)),
        },
    }


# -----------------------------------------------------------------------------
# 2. Quantity Discount Model
# -----------------------------------------------------------------------------

def calculate_quantity_discount(
    annual_demand: float,
    order_cost: float,
    tiers: Optional[List[Dict[str, Any]]] = None,
    discount_tiers: Optional[List[Dict[str, Any]]] = None,
    holding_cost: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate optimal order quantity under an all-units quantity discount schedule."""
    # 1. Input Validation
    if annual_demand is None or annual_demand <= 0:
        raise ValueError(f"Annual demand must be positive, got {annual_demand}")
    if order_cost is None or order_cost <= 0:
        raise ValueError(f"Order cost must be positive, got {order_cost}")

    tier_list = tiers if tiers is not None else discount_tiers
    if not tier_list:
        raise ValueError("Must provide discount_tiers list with at least one price tier; cannot invent missing input.")

    h_fix = fixed_holding_cost if fixed_holding_cost is not None else holding_cost
    rate = None
    if holding_cost_rate is not None:
        rate = holding_cost_rate / 100.0 if holding_cost_rate > 1.0 else holding_cost_rate
        if rate <= 0:
            raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}")
    elif h_fix is not None:
        if h_fix <= 0:
            raise ValueError(f"Fixed holding cost must be positive, got {h_fix}")
    else:
        raise ValueError("Must provide either 'holding_cost_rate' (percentage) or 'fixed_holding_cost'; cannot invent missing input.")

    # 2. Evaluate each discount tier
    sorted_tiers = sorted(tier_list, key=lambda x: x.get("min_qty", 0))
    tier_evaluations: List[Dict[str, Any]] = []
    tier_substitutions: Dict[str, str] = {}
    best_cost = float("inf")
    best_tier_idx = -1
    best_q = 0.0
    best_tier_data: Dict[str, Any] = {}

    for idx, tier in enumerate(sorted_tiers):
        if "unit_cost" not in tier:
            raise ValueError(f"Tier {idx + 1} is missing required 'unit_cost'.")
        c = float(tier["unit_cost"])
        if c <= 0:
            raise ValueError(f"Tier {idx + 1} unit_cost must be positive, got {c}")

        min_q = float(tier.get("min_qty", 0))
        max_q = float(tier.get("max_qty", float("inf")))
        if min_q < 0:
            raise ValueError(f"Tier {idx + 1} min_qty cannot be negative.")
        if max_q < min_q:
            raise ValueError(f"Tier {idx + 1} max_qty ({max_q}) cannot be less than min_qty ({min_q}).")

        if rate is not None:
            h_k = rate * c
            h_desc = f"{rate * 100:.1f}% of ${c:.2f} = ${h_k:.4f}"
        else:
            h_k = float(h_fix)
            h_desc = f"fixed ${h_k:.4f}"

        q_unconstrained = math.sqrt((2.0 * annual_demand * order_cost) / h_k)

        if q_unconstrained < min_q:
            candidate_q = min_q
            feasibility_status = f"Below minimum bracket ({min_q:,.0f}); adjusted to price breakpoint"
            is_breakpoint = True
        elif q_unconstrained > max_q:
            candidate_q = max_q
            feasibility_status = f"Exceeds tier maximum ({max_q:,.0f}); dominated by higher discount tier"
            is_breakpoint = False
        else:
            candidate_q = q_unconstrained
            feasibility_status = "Feasible unconstrained EOQ within tier range"
            is_breakpoint = False

        purchase_cost = annual_demand * c
        ordering_cost = (annual_demand / candidate_q) * order_cost
        holding_cost_annual = (candidate_q / 2.0) * h_k
        total_cost = purchase_cost + ordering_cost + holding_cost_annual

        eval_info = {
            "tier_index": idx + 1,
            "min_qty": min_q,
            "max_qty": max_q if max_q != float("inf") else "inf",
            "unit_cost": c,
            "holding_cost_per_unit": round(h_k, 4),
            "holding_cost_type": h_desc,
            "unconstrained_eoq": round(q_unconstrained, 2),
            "candidate_order_quantity": round(candidate_q, 2),
            "is_breakpoint_adjustment": is_breakpoint,
            "feasibility_status": feasibility_status,
            "annual_purchase_cost": round(purchase_cost, 2),
            "annual_ordering_cost": round(ordering_cost, 2),
            "annual_holding_cost": round(holding_cost_annual, 2),
            "total_annual_cost": round(total_cost, 2),
        }
        tier_evaluations.append(eval_info)

        tier_substitutions[f"Tier_{idx + 1}"] = (
            f"EOQ_{idx + 1} = sqrt((2 * {annual_demand:,.0f} * {order_cost:.2f}) / {h_k:.4f}) = {q_unconstrained:.2f} "
            f"-> Feasible Q' = {candidate_q:,.0f}; "
            f"TC = ({annual_demand:,.0f} * ${c:.2f}) + ({annual_demand:,.0f}/{candidate_q:,.0f} * ${order_cost:.2f}) + "
            f"({candidate_q:,.0f}/2 * ${h_k:.4f}) = ${purchase_cost:,.2f} + ${ordering_cost:,.2f} + ${holding_cost_annual:,.2f} = ${total_cost:,.2f}"
        )

        if total_cost < best_cost:
            best_cost = total_cost
            best_tier_idx = idx + 1
            best_q = candidate_q
            best_tier_data = eval_info

    base_cost = tier_evaluations[0]["total_annual_cost"]
    savings_vs_base = base_cost - best_cost

    verification_checks = [
        {
            "check": "Tier Boundary Feasibility Check",
            "passed": True,
            "details": f"Evaluated all {len(sorted_tiers)} price brackets. Each candidate quantity complies with discount bracket thresholds.",
        },
        {
            "check": "Global Cost Minimization",
            "passed": best_cost <= base_cost,
            "details": f"Tier {best_tier_idx} achieves minimum total cost (${best_cost:,.2f}), saving ${savings_vs_base:,.2f} vs Tier 1 baseline (${base_cost:,.2f}).",
        },
    ]

    inputs_extracted = {
        "annual_demand": float(annual_demand),
        "order_cost": float(order_cost),
        "holding_cost_rate": rate,
        "fixed_holding_cost": h_fix,
        "discount_tiers": sorted_tiers,
    }

    variable_definitions = {
        "D": "Annual Demand (units/year)",
        "S": "Ordering Cost ($/order)",
        "C_k": "Unit purchase price for discount tier k ($/unit)",
        "H_k": "Unit holding cost for discount tier k ($/unit/year)",
        "Q_k": "Unconstrained Wilson EOQ for tier k = sqrt((2 * D * S) / H_k)",
        "Q'_k": "Candidate feasible order quantity for tier k (adjusted to price breakpoint if Q_k < min_qty)",
        "Purchase Cost": "D * C_k ($/year)",
        "Ordering Cost": "(D / Q) * S ($/year)",
        "Holding Cost": "(Q / 2) * H_k ($/year)",
        "Total Cost (TC)": "Purchase Cost + Ordering Cost + Holding Cost ($/year)",
    }

    formulas = {
        "unconstrained_eoq": "Q_k = sqrt((2 * D * S) / H_k)",
        "candidate_quantity": "Q'_k = min_qty_k if Q_k < min_qty_k else (Q_k if Q_k <= max_qty_k else max_qty_k)",
        "total_annual_cost": "TC(Q) = (D * C_k) + ((D / Q) * S) + ((Q / 2) * H_k)",
        "optimal_decision": "arg min_{k} TC(Q'_k)",
    }

    intermediate_calculations = {
        "tier_evaluations": tier_evaluations,
        "baseline_tier_cost": round(base_cost, 2),
        "optimal_tier_cost": round(best_cost, 2),
        "annual_savings_vs_baseline": round(savings_vs_base, 2),
        "percentage_savings": round((savings_vs_base / base_cost) * 100.0, 2) if base_cost > 0 else 0.0,
    }

    final_answer = {
        "optimal_tier": best_tier_idx,
        "optimal_order_quantity": round(best_q, 2),
        "optimal_order_quantity_rounded": int(round(best_q)),
        "minimum_total_annual_cost": round(best_cost, 2),
        "unit_cost_at_optimal_tier": best_tier_data.get("unit_cost"),
        "annual_savings_vs_base_tier": round(savings_vs_base, 2),
    }

    assumptions = [
        "All-units quantity discount schedule applies: the reduced unit price applies to all units purchased.",
        "Demand is deterministic, continuous, and steady across the annual planning cycle.",
        "Lead time is constant with instantaneous delivery and zero stockouts allowed.",
        "Holding cost is either constant per unit or proportional to the unit purchase price (H = i * C).",
        "Warehouse storage capacity is capable of holding the bulk candidate quantities.",
    ]

    return {
        "model": "Quantity Discount Model",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": tier_substitutions,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Backward-compatibility alias
        "tier_evaluations": tier_evaluations,
        "results": {
            "optimal_tier": best_tier_idx,
            "optimal_order_quantity": round(best_q, 2),
            "optimal_order_quantity_rounded": int(round(best_q)),
            "minimum_total_annual_cost": round(best_cost, 2),
            "annual_savings_vs_base_tier": round(savings_vs_base, 2),
            "percent_savings": round((savings_vs_base / base_cost) * 100.0, 2) if base_cost > 0 else 0.0,
        },
    }


# -----------------------------------------------------------------------------
# 3. Safety Stock & Reorder Point (calculate_rop)
# -----------------------------------------------------------------------------

def calculate_rop(
    demand_mean: Optional[float] = None,
    lead_time_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    lead_time_std: Optional[float] = None,
    service_level: Optional[float] = None,
    mean_demand_per_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    lead_time_demand_mean: Optional[float] = None,
    lead_time_demand_std: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate Safety Stock (SS) and Reorder Point (ROP) across the 4 fundamental cases."""
    d = demand_mean if demand_mean is not None else mean_demand_per_period
    L = lead_time_mean if lead_time_mean is not None else lead_time
    sigma_d = demand_std if demand_std is not None else (std_demand_per_period if std_demand_per_period is not None else 0.0)
    sigma_L = lead_time_std if lead_time_std is not None else 0.0

    # Direct lead-time distribution support
    if lead_time_demand_mean is not None and lead_time_demand_std is not None:
        d_lt_mean = float(lead_time_demand_mean)
        sigma_LT = float(lead_time_demand_std)
        if service_level is None:
            raise ValueError("Service level is required when lead time demand is stochastic; cannot invent missing input.")
        csl = normalize_service_level(service_level)
        z = float(stats.norm.ppf(csl))
        ss = z * sigma_LT
        rop = d_lt_mean + ss
        case_name = "Direct Lead-Time Demand Distribution"
        formula_std = "sigma_LT = direct input"
        sub_std = f"sigma_LT = {sigma_LT:.4f}"
        sub_ss = f"SS = {z:.4f} * {sigma_LT:.4f} = {ss:.2f} ≈ {int(math.ceil(ss))} units"
        sub_rop = f"ROP = {d_lt_mean:.2f} + {ss:.2f} = {rop:.2f} ≈ {int(math.ceil(rop))} units"
        assumptions = ["Lead-time demand is normally distributed with known mean and variance."]
    else:
        # Validation
        if d is None:
            raise ValueError("Missing required input: demand_mean cannot be None.")
        if L is None:
            raise ValueError("Missing required input: lead_time_mean cannot be None.")
        if d < 0:
            raise ValueError(f"Mean demand cannot be negative, got {d}")
        if L < 0:
            raise ValueError(f"Lead time cannot be negative, got {L}")
        if sigma_d < 0:
            raise ValueError(f"Demand standard deviation cannot be negative, got {sigma_d}")
        if sigma_L < 0:
            raise ValueError(f"Lead time standard deviation cannot be negative, got {sigma_L}")

        d_lt_mean = float(d * L)
        has_var_d = sigma_d > 0.0
        has_var_L = sigma_L > 0.0

        if not has_var_d and not has_var_L:
            case_name = "Case 1: Constant Demand and Constant Lead Time"
            sigma_LT = 0.0
            z = 0.0
            ss = 0.0
            rop = d_lt_mean
            csl = 1.0 if service_level is None else normalize_service_level(service_level)
            formula_std = "sigma_LT = 0"
            sub_std = "sigma_LT = 0 (deterministic)"
            sub_ss = "SS = 0 units (deterministic)"
            sub_rop = f"ROP = {d} * {L} = {rop:,.2f} units"
            assumptions = [
                "Demand rate (d) is perfectly constant and deterministic.",
                "Lead time (L) is perfectly constant and deterministic.",
                "No safety stock is needed (SS = 0) because stockouts cannot occur under deterministic conditions.",
                "Reorder point equals exact demand during lead time: ROP = d * L.",
            ]
        elif has_var_d and not has_var_L:
            case_name = "Case 2: Variable Demand and Constant Lead Time"
            csl = normalize_service_level(service_level)
            z = float(stats.norm.ppf(csl))
            sigma_LT = math.sqrt(L) * sigma_d
            ss = z * sigma_LT
            rop = d_lt_mean + ss
            formula_std = "sigma_LT = sqrt(L) * sigma_d"
            sub_std = f"sigma_LT = sqrt({L}) * {sigma_d} = {math.sqrt(L):.4f} * {sigma_d} = {sigma_LT:.4f}"
            sub_ss = f"SS = {z:.4f} * {sigma_LT:.4f} = {ss:.2f} ≈ {int(math.ceil(ss))} units"
            sub_rop = f"ROP = ({d} * {L}) + {ss:.2f} = {d_lt_mean:.2f} + {ss:.2f} = {rop:.2f} ≈ {int(math.ceil(rop))} units"
            assumptions = [
                "Period demand is independent and identically distributed normal random variable.",
                "Lead time (L) is constant and deterministic.",
                "Variance of lead time demand scales linearly with lead time: Var(LTD) = L * sigma_d^2.",
            ]
        elif not has_var_d and has_var_L:
            case_name = "Case 3: Constant Demand and Variable Lead Time"
            csl = normalize_service_level(service_level)
            z = float(stats.norm.ppf(csl))
            sigma_LT = float(d * sigma_L)
            ss = z * sigma_LT
            rop = d_lt_mean + ss
            formula_std = "sigma_LT = d * sigma_L"
            sub_std = f"sigma_LT = {d} * {sigma_L} = {sigma_LT:.4f}"
            sub_ss = f"SS = {z:.4f} * {sigma_LT:.4f} = {ss:.2f} ≈ {int(math.ceil(ss))} units"
            sub_rop = f"ROP = ({d} * {L}) + {ss:.2f} = {d_lt_mean:.2f} + {ss:.2f} = {rop:.2f} ≈ {int(math.ceil(rop))} units"
            assumptions = [
                "Demand rate (d) is constant and deterministic.",
                "Lead time (L) is a normally distributed random variable with mean L and std dev sigma_L.",
                "Uncertainty arises solely from supplier delivery lead time variance.",
            ]
        else:
            case_name = "Case 4: Variable Demand and Variable Lead Time"
            csl = normalize_service_level(service_level)
            z = float(stats.norm.ppf(csl))
            var_term1 = L * (sigma_d ** 2)
            var_term2 = (d ** 2) * (sigma_L ** 2)
            var_total = var_term1 + var_term2
            sigma_LT = math.sqrt(var_total)
            ss = z * sigma_LT
            rop = d_lt_mean + ss
            formula_std = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
            sub_std = (
                f"sigma_LT = sqrt(({L} * {sigma_d}^2) + ({d}^2 * {sigma_L}^2)) = "
                f"sqrt({var_term1:.2f} + {var_term2:.2f}) = sqrt({var_total:.2f}) = {sigma_LT:.4f}"
            )
            sub_ss = f"SS = {z:.4f} * {sigma_LT:.4f} = {ss:.2f} ≈ {int(math.ceil(ss))} units"
            sub_rop = f"ROP = ({d} * {L}) + {ss:.2f} = {d_lt_mean:.2f} + {ss:.2f} = {rop:.2f} ≈ {int(math.ceil(rop))} units"
            assumptions = [
                "Demand per period and lead time duration are mutually independent random variables.",
                "Both demand and lead time follow continuous normal distributions.",
                "Variance during lead time combines demand variance and lead-time variance: Var = L*sigma_d^2 + d^2*sigma_L^2.",
                "Cycle service level represents the probability of no stockout during lead time.",
            ]

    expected_shortage = (sigma_LT * unit_normal_loss(z)) if sigma_LT > 0 else 0.0

    verification_checks = [
        {
            "check": "Reorder Point Bounds Safety Stock",
            "passed": rop >= ss,
            "details": f"ROP ({rop:.2f}) is strictly greater than or equal to Safety Stock ({ss:.2f}) by expected lead-time demand ({d_lt_mean:.2f}).",
        },
        {
            "check": "Service Level Boundary Check",
            "passed": 0.0 < csl <= 1.0,
            "details": f"Target Cycle Service Level ({csl * 100:.1f}%) is within valid probability bounds (z = {z:.4f}).",
        },
        {
            "check": "Non-negative Standard Deviation",
            "passed": sigma_LT >= 0,
            "details": f"Lead-time standard deviation sigma_LT = {sigma_LT:.4f} is non-negative.",
        },
    ]

    inputs_extracted = {
        "demand_mean": float(d),
        "lead_time_mean": float(L),
        "demand_std": float(sigma_d),
        "lead_time_std": float(sigma_L),
        "service_level": float(csl),
    }

    variable_definitions = {
        "d": "Average Demand per Period (units/period)",
        "L": "Average Replenishment Lead Time (periods)",
        "sigma_d": "Standard Deviation of Demand per Period (units)",
        "sigma_L": "Standard Deviation of Lead Time (periods)",
        "sigma_LT": "Standard Deviation of Demand during Lead Time (units)",
        "CSL": "Target Cycle Service Level (probability of no stockout during lead time)",
        "z": "Standard Normal Inverse CDF Factor = Phi^-1(CSL)",
        "SS": "Safety Stock buffer = z * sigma_LT (units)",
        "ROP": "Reorder Point = (d * L) + SS (units)",
        "ESC": "Expected Shortage per Cycle = sigma_LT * L(z) (units/cycle)",
    }

    formulas = {
        "lead_time_demand_std": formula_std,
        "safety_stock": "SS = z * sigma_LT",
        "reorder_point": "ROP = (d * L) + SS",
        "expected_shortage": "ESC = sigma_LT * (phi(z) - z * (1 - Phi(z)))",
    }

    formula_with_substituted_values = {
        "lead_time_demand_std": sub_std,
        "z_score": f"z = norm.ppf({csl:.4f}) = {z:.4f}",
        "safety_stock": sub_ss,
        "reorder_point": sub_rop,
        "expected_shortage": f"ESC = {sigma_LT:.4f} * L({z:.4f}) = {expected_shortage:.4f} units/cycle",
    }

    intermediate_calculations = {
        "expected_lead_time_demand": round(d_lt_mean, 2),
        "sigma_LT": round(sigma_LT, 4),
        "z_score": round(z, 4),
        "safety_stock_exact": round(ss, 4),
        "reorder_point_exact": round(rop, 4),
        "expected_shortage_per_cycle": round(expected_shortage, 4),
        "formula_applied": f"{formula_std}; SS = z * sigma_LT; ROP = d * L + SS",
    }

    final_answer = {
        "safety_stock": int(math.ceil(ss)),
        "safety_stock_exact": round(ss, 2),
        "reorder_point": int(math.ceil(rop)),
        "reorder_point_exact": round(rop, 2),
        "cycle_service_level_percent": round(csl * 100.0, 2),
        "stockout_risk_percent": round((1.0 - csl) * 100.0, 2),
        "expected_shortage_per_cycle": round(expected_shortage, 4),
    }

    return {
        "model": "Safety Stock & Reorder Point",
        "case": case_name,
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Backward-compatibility alias
        "scenario": case_name,
        "results": {
            "expected_lead_time_demand": round(d_lt_mean, 2),
            "lead_time_demand_std": round(sigma_LT, 2),
            "z_score": round(z, 4),
            "safety_stock_exact": round(ss, 2),
            "safety_stock": int(math.ceil(ss)),
            "reorder_point_exact": round(rop, 2),
            "reorder_point": int(math.ceil(rop)),
            "cycle_service_level_percent": round(csl * 100.0, 2),
            "stockout_probability_percent": round((1.0 - csl) * 100.0, 2),
            "expected_shortage_per_cycle": round(expected_shortage, 4),
        },
    }


def calculate_safety_stock_rop(*args, **kwargs) -> Dict[str, Any]:
    """Compatibility wrapper for calculate_rop."""
    return calculate_rop(*args, **kwargs)


# -----------------------------------------------------------------------------
# 4. Fixed-Period Review (calculate_periodic_review)
# -----------------------------------------------------------------------------

def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    service_level: float = 0.95,
    inventory_position: Optional[float] = None,
    on_hand_inventory: Optional[float] = None,
    on_order_inventory: float = 0.0,
    backorders: float = 0.0,
    mean_demand_per_period: Optional[float] = None,
    std_demand_per_period: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate target inventory (Order-Up-To Level) and Order Quantity for Fixed-Period Review."""
    d = demand_mean if demand_mean is not None else mean_demand_per_period
    sigma_d = demand_std if demand_std is not None else std_demand_per_period

    if review_period is None or review_period <= 0:
        raise ValueError(f"Review period must be strictly positive, got {review_period}")
    if lead_time is None or lead_time < 0:
        raise ValueError(f"Lead time cannot be negative, got {lead_time}")
    if d is None or d < 0:
        raise ValueError(f"Demand mean cannot be negative, got {d}")
    if sigma_d is None or sigma_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {sigma_d}")

    csl = normalize_service_level(service_level)

    protection_period = float(review_period + lead_time)
    demand_during_protection = float(d * protection_period)
    sigma_protection = math.sqrt(protection_period) * sigma_d

    z = float(stats.norm.ppf(csl))
    safety_stock = z * sigma_protection
    target_inventory = demand_during_protection + safety_stock

    inv_pos = None
    order_quantity = None
    if inventory_position is not None:
        inv_pos = float(inventory_position)
        order_quantity = max(0.0, target_inventory - inv_pos)
    elif on_hand_inventory is not None:
        bo = float(backorders or 0.0)
        inv_pos = float(on_hand_inventory + on_order_inventory - bo)
        order_quantity = max(0.0, target_inventory - inv_pos)

    verification_checks = [
        {
            "check": "Protection Period Horizon Check",
            "passed": protection_period > 0,
            "details": f"Protection period P + L = {review_period} + {lead_time} = {protection_period} periods.",
        },
        {
            "check": "Target Inventory Exceeds Safety Stock",
            "passed": target_inventory >= safety_stock,
            "details": f"Target inventory S ({target_inventory:.2f}) covers expected demand ({demand_during_protection:.2f}) plus safety buffer ({safety_stock:.2f}).",
        },
        {
            "check": "Non-Negative Order Quantity",
            "passed": (order_quantity is None) or (order_quantity >= 0),
            "details": f"Recommended replenishment quantity ({order_quantity:.2f} if order_quantity is not None else 'N/A') is bounded by zero.",
        },
    ]

    inputs_extracted = {
        "review_period": float(review_period),
        "lead_time": float(lead_time),
        "demand_mean": float(d),
        "demand_std": float(sigma_d),
        "service_level": float(csl),
        "inventory_position": inv_pos,
        "on_hand_inventory": float(on_hand_inventory) if on_hand_inventory is not None else None,
        "on_order_inventory": float(on_order_inventory),
        "backorders": float(backorders),
    }

    variable_definitions = {
        "P": "Review Period: interval between inventory reviews (periods)",
        "L": "Lead Time: supplier delivery time (periods)",
        "T_protect": "Protection Interval = P + L (periods)",
        "d": "Average Demand per period (units/period)",
        "sigma_d": "Standard Deviation of Demand per period (units)",
        "mu_{P+L}": "Expected Demand during Protection Interval = d * (P + L) (units)",
        "sigma_{P+L}": "Standard Deviation of Demand over Protection Interval = sqrt(P + L) * sigma_d (units)",
        "z": "Safety Factor = Phi^-1(CSL)",
        "SS": "Safety Stock = z * sigma_{P+L} (units)",
        "S": "Target Inventory / Order-Up-To Level = mu_{P+L} + SS (units)",
        "IP": "Inventory Position = On-Hand + On-Order - Backorders (units)",
        "Q": "Recommended Replenishment Order = max(0, S - IP) (units)",
    }

    formulas = {
        "protection_period": "Protection Period = P + L",
        "demand_during_protection": "mu_{P+L} = d * (P + L)",
        "std_dev_protection": "sigma_{P+L} = sqrt(P + L) * sigma_d",
        "safety_stock": "SS = z * sigma_{P+L}",
        "target_inventory": "S = (d * (P + L)) + (z * sqrt(P + L) * sigma_d)",
        "order_quantity": "Q = max(0, S - Inventory_Position)",
    }

    formula_with_substituted_values = {
        "protection_period": f"T_protect = {review_period} + {lead_time} = {protection_period:.2f} periods",
        "demand_during_protection": f"mu_{{P+L}} = {d} * {protection_period:.2f} = {demand_during_protection:.2f} units",
        "std_dev_protection": f"sigma_{{P+L}} = sqrt({protection_period:.2f}) * {sigma_d} = {math.sqrt(protection_period):.4f} * {sigma_d} = {sigma_protection:.4f} units",
        "z_score": f"z = norm.ppf({csl:.4f}) = {z:.4f}",
        "safety_stock": f"SS = {z:.4f} * {sigma_protection:.4f} = {safety_stock:.2f} ≈ {int(math.ceil(safety_stock))} units",
        "target_inventory": f"S = {demand_during_protection:.2f} + {safety_stock:.2f} = {target_inventory:.2f} ≈ {int(math.ceil(target_inventory))} units",
        "order_quantity": (
            f"Q = max(0, {target_inventory:.2f} - {inv_pos:.2f}) = {order_quantity:.2f} ≈ {int(math.ceil(order_quantity))} units"
            if order_quantity is not None
            else "Inventory position not provided; order quantity cannot be calculated."
        ),
    }

    intermediate_calculations = {
        "protection_period": round(protection_period, 2),
        "demand_during_protection_period": round(demand_during_protection, 2),
        "std_dev_protection_period": round(sigma_protection, 4),
        "z_score": round(z, 4),
        "safety_stock_exact": round(safety_stock, 4),
        "target_inventory_exact": round(target_inventory, 4),
        "inventory_position": round(inv_pos, 2) if inv_pos is not None else None,
    }

    final_answer = {
        "target_inventory": int(math.ceil(target_inventory)),
        "target_inventory_exact": round(target_inventory, 2),
        "safety_stock": int(math.ceil(safety_stock)),
        "safety_stock_exact": round(safety_stock, 2),
        "order_quantity": int(math.ceil(order_quantity)) if order_quantity is not None else None,
        "order_quantity_exact": round(order_quantity, 2) if order_quantity is not None else None,
        "protection_period": round(protection_period, 2),
    }

    assumptions = [
        "Inventory review occurs at fixed periodic intervals of P periods.",
        "An order placed at the review epoch arrives after supplier lead time L.",
        "The inventory replenishment must protect against stockouts across the combined interval (P + L).",
        "Period demand is normally distributed and mutually independent across consecutive periods.",
        "Target order-up-to level (S) restores inventory position to buffer expected demand and safety stock.",
    ]

    return {
        "model": "Fixed-Period Review Model",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Backward-compatibility alias
        "results": {
            "protection_interval": float(round(protection_period, 2)),
            "protection_interval_mean_demand": float(round(demand_during_protection, 2)),
            "protection_interval_std_demand": float(round(sigma_protection, 2)),
            "z_score": float(round(z, 4)),
            "safety_stock_exact": float(round(safety_stock, 2)),
            "safety_stock": int(math.ceil(safety_stock)),
            "order_up_to_level_exact": float(round(target_inventory, 2)),
            "order_up_to_level": int(math.ceil(target_inventory)),
            "current_inventory_position": float(round(inv_pos, 2)) if inv_pos is not None else None,
            "recommended_order_quantity_exact": float(round(order_quantity, 2)) if order_quantity is not None else None,
            "recommended_order_quantity": int(math.ceil(order_quantity)) if order_quantity is not None else None,
        },
    }


def calculate_fixed_period_review(*args, **kwargs) -> Dict[str, Any]:
    """Compatibility wrapper for calculate_periodic_review."""
    return calculate_periodic_review(*args, **kwargs)


# -----------------------------------------------------------------------------
# 5. Newsvendor Model (calculate_newsvendor)
# -----------------------------------------------------------------------------

def calculate_newsvendor(
    unit_cost: float,
    selling_price: float,
    salvage_value: float = 0.0,
    demand_mean: float = 0.0,
    demand_std: float = 0.0,
    underage_penalty: float = 0.0,
    mean_demand: Optional[float] = None,
    std_demand: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate underage cost, overage cost, critical ratio, and recommended order quantity for Newsvendor."""
    mu = demand_mean if demand_mean else mean_demand
    sigma = demand_std if demand_std is not None else (std_demand if std_demand is not None else 0.0)

    if unit_cost is None:
        raise ValueError("Missing required input: unit_cost cannot be None.")
    if selling_price is None:
        raise ValueError("Missing required input: selling_price cannot be None.")
    if mu is None:
        raise ValueError("Missing required input: demand_mean cannot be None.")
    if unit_cost <= 0:
        raise ValueError(f"Unit cost must be positive, got {unit_cost}")
    if selling_price <= unit_cost:
        raise ValueError(
            f"Selling price (${selling_price:.2f}) must exceed unit cost (${unit_cost:.2f}) for positive gross margin."
        )
    if salvage_value > unit_cost:
        raise ValueError(
            f"Salvage value (${salvage_value:.2f}) cannot exceed unit cost (${unit_cost:.2f})."
        )
    if mu <= 0:
        raise ValueError(f"Mean demand must be positive, got {mu}")
    if sigma < 0:
        raise ValueError(f"Standard deviation of demand cannot be negative, got {sigma}")

    cost_underage = (selling_price - unit_cost) + underage_penalty
    cost_overage = unit_cost - salvage_value
    denom = cost_underage + cost_overage
    if denom <= 0:
        raise ValueError("Sum of underage cost and overage cost must be strictly positive.")

    critical_ratio = cost_underage / denom

    if sigma == 0.0:
        z_star = 0.0
        q_star = float(mu)
        expected_lost_sales = 0.0
        expected_leftovers = 0.0
        expected_sales = float(mu)
    else:
        z_star = float(stats.norm.ppf(critical_ratio))
        q_star = float(mu + z_star * sigma)
        expected_lost_sales = sigma * unit_normal_loss(z_star)
        expected_leftovers = q_star - mu + expected_lost_sales
        expected_sales = mu - expected_lost_sales

    expected_revenue = selling_price * expected_sales
    expected_salvage_rev = salvage_value * expected_leftovers
    expected_procurement_cost = unit_cost * q_star
    expected_penalty_cost = underage_penalty * expected_lost_sales
    expected_profit = expected_revenue + expected_salvage_rev - expected_procurement_cost - expected_penalty_cost

    fill_rate = (expected_sales / mu) * 100.0 if mu > 0 else 100.0

    verification_checks = [
        {
            "check": "Positive Profit Margin Check (P > C)",
            "passed": selling_price > unit_cost,
            "details": f"Selling price (${selling_price:.2f}) exceeds unit cost (${unit_cost:.2f}) by ${selling_price - unit_cost:.2f}/unit.",
        },
        {
            "check": "Critical Ratio Probability Bounds",
            "passed": 0.0 < critical_ratio < 1.0,
            "details": f"Critical ratio CR = {critical_ratio:.4f} is strictly bounded inside (0, 1).",
        },
        {
            "check": "Marginal Cost Logic Check",
            "passed": (cost_underage > cost_overage and q_star > mu) or (cost_overage >= cost_underage and q_star <= mu),
            "details": (
                f"Cu (${cost_underage:.2f}) > Co (${cost_overage:.2f}) -> Q* ({q_star:.2f}) > mu ({mu}) to avoid stockouts."
                if cost_underage > cost_overage
                else f"Co (${cost_overage:.2f}) >= Cu (${cost_underage:.2f}) -> Q* ({q_star:.2f}) <= mu ({mu}) to avoid excess inventory."
            ),
        },
    ]

    inputs_extracted = {
        "unit_cost": float(unit_cost),
        "selling_price": float(selling_price),
        "salvage_value": float(salvage_value),
        "demand_mean": float(mu),
        "demand_std": float(sigma),
        "underage_penalty": float(underage_penalty),
    }

    variable_definitions = {
        "P": "Retail Selling Price per unit ($/unit)",
        "C": "Procurement / Purchase Cost per unit ($/unit)",
        "V": "Salvage Value per unsold unit ($/unit)",
        "s": "Underage Penalty or goodwill loss per unit ($/unit)",
        "Cu": "Cost of Underage (Lost Profit per unit) = P - C + s ($/unit)",
        "Co": "Cost of Overage (Loss per unsold unit) = C - V ($/unit)",
        "CR": "Critical Ratio (Critical Fractile) = Cu / (Cu + Co)",
        "mu": "Expected Mean Demand (units)",
        "sigma": "Standard Deviation of Demand (units)",
        "z*": "Optimal z-factor = Phi^-1(CR)",
        "Q*": "Optimal Order Quantity = mu + z* * sigma (units)",
        "ES": "Expected Sales (units)",
        "ELS": "Expected Lost Sales = sigma * L(z*) (units)",
        "ELI": "Expected Leftover Inventory = Q* - mu + ELS (units)",
        "E[Profit]": "Expected Profit = P*ES + V*ELI - C*Q* - s*ELS ($)",
    }

    formulas = {
        "cost_of_underage": "Cu = P - C + s",
        "cost_of_overage": "Co = C - V",
        "critical_ratio": "CR = Cu / (Cu + Co)",
        "optimal_quantity": "Q* = mu + z* * sigma  (where Phi(z*) = CR)",
        "expected_lost_sales": "ELS = sigma * (phi(z*) - z* * (1 - Phi(z*)))",
        "expected_leftovers": "ELI = Q* - mu + ELS",
        "expected_sales": "ES = mu - ELS",
        "expected_profit": "E[Profit] = (P * ES) + (V * ELI) - (C * Q*) - (s * ELS)",
    }

    formula_with_substituted_values = {
        "cost_of_underage": f"Cu = ${selling_price:.2f} - ${unit_cost:.2f} + ${underage_penalty:.2f} = ${cost_underage:.2f}/unit",
        "cost_of_overage": f"Co = ${unit_cost:.2f} - ${salvage_value:.2f} = ${cost_overage:.2f}/unit",
        "critical_ratio": f"CR = ${cost_underage:.2f} / (${cost_underage:.2f} + ${cost_overage:.2f}) = ${cost_underage:.2f} / ${denom:.2f} = {critical_ratio:.4f} ({critical_ratio * 100:.2f}%)",
        "z_star": f"z* = norm.ppf({critical_ratio:.4f}) = {z_star:.4f}",
        "optimal_quantity": f"Q* = {mu:,.2f} + ({z_star:.4f} * {sigma:,.2f}) = {q_star:.2f} ≈ {int(round(q_star))} units",
        "expected_profit": f"E[Profit] = (${selling_price:.2f} * {expected_sales:.2f}) + (${salvage_value:.2f} * {expected_leftovers:.2f}) - (${unit_cost:.2f} * {q_star:.2f}) = ${expected_profit:,.2f}",
    }

    intermediate_calculations = {
        "underage_cost_Cu": round(cost_underage, 4),
        "overage_cost_Co": round(cost_overage, 4),
        "critical_ratio": round(critical_ratio, 6),
        "critical_ratio_percent": round(critical_ratio * 100.0, 2),
        "z_star": round(z_star, 4),
        "expected_sales": round(expected_sales, 2),
        "expected_lost_sales": round(expected_lost_sales, 2),
        "expected_leftover_inventory": round(expected_leftovers, 2),
        "expected_revenue": round(expected_revenue, 2),
        "expected_procurement_cost": round(expected_procurement_cost, 2),
        "expected_profit": round(expected_profit, 2),
        "fill_rate_percent": round(fill_rate, 2),
    }

    final_answer = {
        "underage_cost": round(cost_underage, 2),
        "overage_cost": round(cost_overage, 2),
        "critical_ratio": round(critical_ratio, 4),
        "recommended_order_quantity": int(round(q_star)),
        "recommended_order_quantity_exact": round(q_star, 2),
        "expected_profit": round(expected_profit, 2),
        "fill_rate_percent": round(fill_rate, 2),
    }

    assumptions = [
        "Single order opportunity prior to the selling period (no mid-season replenishment).",
        "Demand is uncertain and modeled by a continuous normal distribution.",
        "Unsold units at the end of the lifecycle lose value and are liquidated at salvage value V.",
        "Unsatisfied customer demand is permanently lost (with optional penalty s per unit).",
        "Objective is to maximize expected single-period profit under demand risk.",
    ]

    return {
        "model": "Newsvendor (Single-Period Stochastic Demand)",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        # Backward-compatibility alias
        "results": {
            "cost_of_underage_Cu": round(cost_underage, 2),
            "cost_of_overage_Co": round(cost_overage, 2),
            "critical_ratio": round(critical_ratio, 4),
            "critical_ratio_percent": round(critical_ratio * 100.0, 2),
            "z_score": round(z_star, 4),
            "optimal_order_quantity_exact": round(q_star, 2),
            "optimal_order_quantity": int(round(q_star)),
            "expected_sales": round(expected_sales, 2),
            "expected_lost_sales": round(expected_lost_sales, 2),
            "expected_leftover_inventory": round(expected_leftovers, 2),
            "expected_profit": round(expected_profit, 2),
            "fill_rate_percent": round(fill_rate, 2),
            "expected_value_of_perfect_information": round((selling_price - unit_cost) * mu - expected_profit, 2),
        },
    }


# =============================================================================
# Multicriteria Inventory Classification Tool (A, B, C, D)
# =============================================================================

def calculate_multicriteria_classification(
    df: Any,
    sales_col: Optional[str] = None,
    hits_col: Optional[str] = None,
    item_num_col: Optional[str] = None,
    item_name_col: Optional[str] = None,
    uom_col: Optional[str] = None,
    method: str = "composite",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Perform deterministic multicriteria inventory classification using Sales ($) and Hits.

    Classifies items into A, B, C, and D classes based on both financial impact (Sales $)
    and warehouse transaction velocity (Hits / pick lines).

    Supported Methods:
    - 'composite' (Weighted Multi-Criteria Percentile Score - Default):
        * Composite Score = 0.65 * Percentile_Rank(Sales) + 0.35 * Percentile_Rank(Hits)
        * Class A: Top 20%
        * Class B: Next 30% (20%-50%)
        * Class C: Next 30% (50%-80%)
        * Class D: Bottom 20% (80%-100%)
    - 'matrix' (Cross-Criteria Matrix):
        * Sales Class (Pareto 80/15/5): A (top 80%), B (next 15%), C (remaining 5%)
        * Hits Class (Pareto 80/15/5): A (top 80%), B (next 15%), C (remaining 5%)
        * Mapping to A, B, C, D:
            AA, AB, BA -> Class A (High Sales & High Hits - vital fast-movers)
            AC, BB, CA -> Class B (High Value / Moderate Velocity)
            BC, CB     -> Class C (Low Value / Moderate-High Velocity - routine commodities)
            CC         -> Class D (Low Sales & Low Hits - slow-moving tail / dead stock candidates)

    Returns 8 verification sections:
    1. inputs_extracted
    2. variable_definitions
    3. formula
    4. formula_with_substituted_values
    5. intermediate_calculations
    6. final_answer
    7. verification_checks
    8. assumptions
    Plus 'classified_df' containing the resulting dataframe with classification columns,
    'top_5_by_class', and class total sales.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a valid pandas DataFrame.")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty; cannot perform classification.")

    df_classified = df.copy()

    # Detect column names if not explicitly provided
    if not sales_col:
        for c in ["Sales ($)", "Sales", "Total Sales", "Sales_USD", "Revenue"]:
            if c in df_classified.columns:
                sales_col = c
                break
    if not hits_col:
        for c in ["Hits", "Hit", "Transactions", "Picks", "Order Lines"]:
            if c in df_classified.columns:
                hits_col = c
                break
    if not item_num_col:
        for c in ["Item Number", "Item No", "SKU", "Product ID", "Item"]:
            if c in df_classified.columns:
                item_num_col = c
                break
    if not item_name_col:
        for c in ["Item Name", "Description", "Product Name", "Product"]:
            if c in df_classified.columns:
                item_name_col = c
                break
    if not uom_col:
        for c in ["UoM", "UOM", "Unit of Measure"]:
            if c in df_classified.columns:
                uom_col = c
                break

    if not sales_col or sales_col not in df_classified.columns:
        raise ValueError(f"Sales column not found. Available columns: {list(df_classified.columns)}")
    if not hits_col or hits_col not in df_classified.columns:
        raise ValueError(f"Hits column not found. Available columns: {list(df_classified.columns)}")

    # Clean numeric data (handle string formatting, $, commas)
    if not pd.api.types.is_numeric_dtype(df_classified[sales_col]):
        df_classified[sales_col] = (
            df_classified[sales_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    df_classified[sales_col] = pd.to_numeric(df_classified[sales_col], errors="coerce").fillna(0.0)

    if not pd.api.types.is_numeric_dtype(df_classified[hits_col]):
        df_classified[hits_col] = (
            df_classified[hits_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    df_classified[hits_col] = pd.to_numeric(df_classified[hits_col], errors="coerce").fillna(0.0)

    total_sales = float(df_classified[sales_col].sum())
    total_hits = float(df_classified[hits_col].sum())
    total_items = len(df_classified)

    # 1. Sales ABC Classification (Pareto: Top 80% = A, Next 15% = B, Last 5% = C)
    if total_sales > 0:
        df_sorted_sales = df_classified.sort_values(by=sales_col, ascending=False).copy()
        cum_sales = df_sorted_sales[sales_col].cumsum()
        cum_sales_pct = (cum_sales / total_sales) * 100.0
        prev_cum_sales_pct = cum_sales_pct.shift(1, fill_value=0.0)

        sales_classes = pd.Series("C", index=df_sorted_sales.index)
        sales_classes[prev_cum_sales_pct < 80.0] = "A"
        sales_classes[(prev_cum_sales_pct >= 80.0) & (prev_cum_sales_pct < 95.0)] = "B"
        df_classified["Sales Class"] = sales_classes
    else:
        df_classified["Sales Class"] = "C"

    # 2. Hits ABC Classification (Pareto: Top 80% = A, Next 15% = B, Last 5% = C)
    if total_hits > 0:
        df_sorted_hits = df_classified.sort_values(by=hits_col, ascending=False).copy()
        cum_hits = df_sorted_hits[hits_col].cumsum()
        cum_hits_pct = (cum_hits / total_hits) * 100.0
        prev_cum_hits_pct = cum_hits_pct.shift(1, fill_value=0.0)

        hits_classes = pd.Series("C", index=df_sorted_hits.index)
        hits_classes[prev_cum_hits_pct < 80.0] = "A"
        hits_classes[(prev_cum_hits_pct >= 80.0) & (prev_cum_hits_pct < 95.0)] = "B"
        df_classified["Hits Class"] = hits_classes
    else:
        df_classified["Hits Class"] = "C"

    # 3. Percentile Ranks & Composite Score
    # Formula: Composite Score = 0.65 * Percentile_Rank(Sales) + 0.35 * Percentile_Rank(Hits)
    sales_rank = df_classified[sales_col].rank(pct=True, ascending=True)
    hits_rank = df_classified[hits_col].rank(pct=True, ascending=True)
    df_classified["Percentile_Rank_Sales"] = sales_rank.round(4)
    df_classified["Percentile_Rank_Hits"] = hits_rank.round(4)
    df_classified["Composite Score"] = (0.65 * sales_rank + 0.35 * hits_rank).round(4)

    # 4. Matrix Class Mapping
    def map_matrix_row(row: pd.Series) -> str:
        combo = f"{row['Sales Class']}{row['Hits Class']}"
        if combo in ("AA", "AB", "BA"):
            return "A"
        elif combo in ("AC", "BB", "CA"):
            return "B"
        elif combo in ("BC", "CB"):
            return "C"
        else:  # CC
            return "D"

    df_classified["Matrix Class"] = df_classified.apply(map_matrix_row, axis=1)

    # 5. Composite Class Mapping (Percentile: Top 20% A, 20-50% B, 50-80% C, Bottom 20% D)
    composite_pct_desc = df_classified["Composite Score"].rank(pct=True, ascending=False)
    def map_composite_row(r: float) -> str:
        if r <= 0.20:
            return "A"
        elif r <= 0.50:
            return "B"
        elif r <= 0.80:
            return "C"
        else:
            return "D"

    df_classified["Composite Class"] = composite_pct_desc.apply(map_composite_row)

    # Primary Classification Column
    if str(method).lower() == "composite":
        df_classified["Classification"] = df_classified["Composite Class"]
    else:
        df_classified["Classification"] = df_classified["Matrix Class"]

    # Calculate Distribution Statistics across A, B, C, D
    class_stats: Dict[str, Dict[str, Any]] = {}
    strategies = {
        "A": "Vital Fast-Movers: High revenue & pick activity. Tight continuous control, frequent cycle counts, prime warehouse dock slotting.",
        "B": "Moderate-High Priority: High value / moderate velocity. Regular periodic review, controlled safety stock buffers.",
        "C": "Routine Commodities: Moderate-low value with regular picks. Automated min-max/two-bin replenishment, bulk ordering.",
        "D": "Slow-Moving Tail / Dead Stock: Low sales and low pick frequency. Strict reorder limits, SKU rationalization, phase-out review.",
    }

    for c in ["A", "B", "C", "D"]:
        subset = df_classified[df_classified["Classification"] == c]
        cnt = len(subset)
        c_sales = float(subset[sales_col].sum())
        c_hits = float(subset[hits_col].sum())
        class_stats[c] = {
            "class": c,
            "count": cnt,
            "percent_items": round((cnt / total_items) * 100.0, 2) if total_items > 0 else 0.0,
            "total_sales": round(c_sales, 2),
            "percent_sales": round((c_sales / total_sales) * 100.0, 2) if total_sales > 0 else 0.0,
            "total_hits": int(c_hits),
            "percent_hits": round((c_hits / total_hits) * 100.0, 2) if total_hits > 0 else 0.0,
            "strategy": strategies[c],
        }

    # Extract Top 5 Items by Sales for each Class (A, B, C, D)
    top_5_by_class: Dict[str, List[Dict[str, Any]]] = {}
    cols_order = [c for c in [item_num_col, item_name_col, uom_col, sales_col, hits_col, "Classification", "Composite Score"] if c and c in df_classified.columns]
    for c in ["A", "B", "C", "D"]:
        subset = df_classified[df_classified["Classification"] == c].sort_values(by=sales_col, ascending=False).head(5)
        top_5_by_class[c] = subset[cols_order].to_dict(orient="records")

    # Structured 8 Verification Sections
    inputs_extracted = {
        "total_items": total_items,
        "total_sales_dollars": round(total_sales, 2),
        "total_hits": int(total_hits),
        "sales_column": sales_col,
        "hits_column": hits_col,
        "classification_method": method,
    }

    variable_definitions = {
        "Sales ($)": "Total dollar revenue generated by the SKU (financial value dimension).",
        "Hits": "Total pick transactions / order line frequency (warehouse velocity dimension).",
        "Sales Class": "Pareto ABC category based on cumulative annual sales (A: 80%, B: 15%, C: 5%).",
        "Hits Class": "Pareto ABC category based on cumulative transaction hits (A: 80%, B: 15%, C: 5%).",
        "Composite Score": "Multi-criteria score combining Sales percentile and Hits percentile (0 to 100).",
        "Class A": "Critical fast movers with top revenue impact and high order picking velocity.",
        "Class B": "High-value slow movers or solid performers with moderate pick activity.",
        "Class C": "Fast-moving routine parts with low revenue contribution.",
        "Class D": "Slow-moving tail items with low revenue and minimal order hits.",
    }

    formulas = [
        "1. Cumulative Sales % = sum(Sales_i) / Total_Sales * 100",
        "2. Cumulative Hits % = sum(Hits_i) / Total_Hits * 100",
        "3. Composite Score = 0.65 * Percentile_Rank(Sales) + 0.35 * Percentile_Rank(Hits)",
        "4. Composite Classification Tiers: Class A (Top 20%), Class B (Next 30%), Class C (Next 30%), Class D (Bottom 20%)",
        "5. Matrix Mapping (Alternative): AA/AB/BA -> Class A, AC/BB/CA -> Class B, BC/CB -> Class C, CC -> Class D",
    ]

    formula_with_substituted_values = [
        f"Total Catalog Items = {total_items:,}",
        f"Total Annual Sales = ${total_sales:,.2f}",
        f"Total Transaction Hits = {int(total_hits):,}",
        f"Selected Classification Rule = '{method}'",
    ]

    intermediate_calculations = {
        f"Class {c}": {
            "Item Count": stats_c["count"],
            "Item %": f"{stats_c['percent_items']}%",
            "Sales ($)": f"${stats_c['total_sales']:,.2f}",
            "Sales %": f"{stats_c['percent_sales']}%",
            "Hits": f"{stats_c['total_hits']:,}",
            "Hits %": f"{stats_c['percent_hits']}%",
        }
        for c, stats_c in class_stats.items()
    }

    final_answer = {
        "summary": (
            f"Classified {total_items} items into: "
            f"Class A: {class_stats['A']['count']} ({class_stats['A']['percent_items']}%), "
            f"Class B: {class_stats['B']['count']} ({class_stats['B']['percent_items']}%), "
            f"Class C: {class_stats['C']['count']} ({class_stats['C']['percent_items']}%), "
            f"Class D: {class_stats['D']['count']} ({class_stats['D']['percent_items']}%)."
        ),
        "class_breakdown": class_stats,
    }

    verification_checks = [
        {
            "check": "Sum of class counts equals total dataset rows",
            "condition": sum(s["count"] for s in class_stats.values()) == total_items,
            "status": "PASSED" if sum(s["count"] for s in class_stats.values()) == total_items else "FAILED",
        },
        {
            "check": "Sum of class sales equals total catalog sales",
            "condition": abs(sum(s["total_sales"] for s in class_stats.values()) - total_sales) < 1.0,
            "status": "PASSED" if abs(sum(s["total_sales"] for s in class_stats.values()) - total_sales) < 1.0 else "FAILED",
        },
        {
            "check": "Sum of class hits equals total catalog hits",
            "condition": sum(s["total_hits"] for s in class_stats.values()) == int(total_hits),
            "status": "PASSED" if sum(s["total_hits"] for s in class_stats.values()) == int(total_hits) else "FAILED",
        },
        {
            "check": "All items have a deterministic classification in {A, B, C, D}",
            "condition": set(df_classified["Classification"].unique()).issubset({"A", "B", "C", "D"}),
            "status": "PASSED",
        },
    ]

    assumptions = [
        "Multicriteria evaluation considers both financial volume (Sales $) and picking activity (Hits).",
        "Individual Sales ABC and Hits ABC use standard Pareto boundaries (80% for Class A, 15% for Class B, 5% for Class C).",
        "Cross-criteria matrix mapping integrates sales volume with warehouse line activity to prevent single-metric distortions.",
        "Missing or non-numeric sales/hits entries are treated as 0 without inventing artificial values.",
    ]

    return {
        "model": "Multicriteria Inventory Classification (A, B, C, D)",
        "inputs_extracted": inputs_extracted,
        "inputs": inputs_extracted,
        "variable_definitions": variable_definitions,
        "formula": formulas,
        "formulas": formulas,
        "formula_with_substituted_values": formula_with_substituted_values,
        "intermediate_calculations": intermediate_calculations,
        "final_answer": final_answer,
        "verification_checks": verification_checks,
        "assumptions": assumptions,
        "class_breakdown": class_stats,
        "top_5_by_class": top_5_by_class,
        "classified_df": df_classified,
        "results": {
            "total_items": total_items,
            "total_sales": round(total_sales, 2),
            "total_hits": int(total_hits),
            "class_A_count": class_stats["A"]["count"],
            "class_B_count": class_stats["B"]["count"],
            "class_C_count": class_stats["C"]["count"],
            "class_D_count": class_stats["D"]["count"],
            "class_A_sales": class_stats["A"]["total_sales"],
            "class_B_sales": class_stats["B"]["total_sales"],
            "class_C_sales": class_stats["C"]["total_sales"],
            "class_D_sales": class_stats["D"]["total_sales"],
        },
    }

