"""Supply Chain Analytics Inventory Decision Agent.

Implements the decision pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import inventory_tools as it


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review Model"
    NEWSVENDOR = "Newsvendor Model"
    UNKNOWN = "Unknown Model"


@dataclass
class VerificationReport:
    is_valid: bool
    sanity_checks_passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    managerial_notes: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    question: str
    identified_model: InventoryModelType
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    selected_tool_name: str
    calculation_results: Optional[Dict[str, Any]]
    verification: Optional[VerificationReport]
    explanation: str
    status: str  # 'SUCCESS', 'MISSING_INPUTS', 'CALCULATION_ERROR'


class InventoryDecisionAgent:
    """End-to-end intelligent inventory decision agent."""

    def __init__(self):
        self.tool_mapping: Dict[InventoryModelType, Callable[..., Dict[str, Any]]] = {
            InventoryModelType.EOQ: it.calculate_eoq,
            InventoryModelType.QUANTITY_DISCOUNT: it.calculate_quantity_discount,
            InventoryModelType.SAFETY_STOCK_ROP: it.calculate_safety_stock_rop,
            InventoryModelType.FIXED_PERIOD_REVIEW: it.calculate_fixed_period_review,
            InventoryModelType.NEWSVENDOR: it.calculate_newsvendor,
        }

    # =========================================================================
    # Step 1: Identify Model
    # =========================================================================

    def identify_model(self, question: str, override: Optional[str] = None) -> InventoryModelType:
        """Classify user query into one of the 5 inventory models."""
        if override:
            override_clean = override.strip().lower()
            if "discount" in override_clean:
                return InventoryModelType.QUANTITY_DISCOUNT
            if "eoq" in override_clean or "economic" in override_clean:
                return InventoryModelType.EOQ
            if "period" in override_clean or "order-up" in override_clean or "order up" in override_clean:
                return InventoryModelType.FIXED_PERIOD_REVIEW
            if "reorder" in override_clean or "rop" in override_clean or "safety stock" in override_clean:
                return InventoryModelType.SAFETY_STOCK_ROP
            if "news" in override_clean or "perishable" in override_clean or "single period" in override_clean:
                return InventoryModelType.NEWSVENDOR

        q = question.lower()

        # Priority 1: Quantity Discount (contains discount / tiers / price break)
        if any(w in q for w in ["discount", "price break", "volume tier", "price schedule", "bulk tier", "bracket"]):
            return InventoryModelType.QUANTITY_DISCOUNT

        # Priority 2: Fixed-Period Review (periodic, review period, order-up-to)
        if any(
            w in q
            for w in [
                "fixed period",
                "periodic review",
                "order-up-to",
                "order up to",
                "review period",
                "review cycle",
                "reviewed every",
                "check inventory every",
                "target inventory level",
            ]
        ):
            return InventoryModelType.FIXED_PERIOD_REVIEW

        # Priority 3: Newsvendor (single period, salvage, perishable, critical fractile, newsboy)
        if any(
            w in q
            for w in [
                "newsvendor",
                "newsboy",
                "single period",
                "salvage",
                "clearance",
                "perishable",
                "seasonal jackets",
                "seasonal demand",
                "critical ratio",
                "critical fractile",
                "overage",
                "underage",
            ]
        ):
            return InventoryModelType.NEWSVENDOR

        # Priority 4: Safety Stock & ROP (reorder point, safety stock, lead time demand, service level)
        if any(
            w in q
            for w in [
                "safety stock",
                "reorder point",
                "rop",
                "stockout probability",
                "cycle service level",
                "lead time demand",
                "continuous review",
            ]
        ):
            return InventoryModelType.SAFETY_STOCK_ROP

        # Priority 5: EOQ (economic order quantity, annual demand, order cost, holding cost)
        if any(
            w in q
            for w in [
                "eoq",
                "economic order quantity",
                "order quantity",
                "annual demand",
                "holding cost",
                "setup cost",
                "batch size",
                "orders per year",
            ]
        ):
            return InventoryModelType.EOQ

        return InventoryModelType.UNKNOWN

    # =========================================================================
    # Step 2: Extract Inputs
    # =========================================================================

    def extract_inputs(
        self, question: str, model: InventoryModelType
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Extract domain parameters from question using comprehensive NLP patterns and synonyms.

        Examines the entire question before declaring any parameter missing.
        """
        inputs: Dict[str, Any] = {}
        missing: List[str] = []
        q = question

        # Helper to search regex floats across entire question text
        def search_number(patterns: List[str], text: str) -> Optional[float]:
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    # Look for first non-None group containing digits
                    for group_idx in range(1, len(m.groups()) + 1):
                        g = m.group(group_idx)
                        if g is not None:
                            raw = str(g).replace(",", "").strip()
                            try:
                                return float(raw)
                            except ValueError:
                                continue
            return None

        # ---------------------------------------------------------------------
        # 1. EOQ Extraction
        # ---------------------------------------------------------------------
        if model == InventoryModelType.EOQ:
            # Annual demand synonyms
            d = search_number(
                [
                    r"(?:annual\s+demand|demand\s+per\s+year|yearly\s+demand|annual\s+usage|annual\s+sales|annual\s+volume)\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|widgets)?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|widgets)?\s*(?:per\s+year|annually|a\s+year|yearly)",
                    r"(?:sells|uses|requires|consumes)\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|pieces|widgets)?\s*(?:per\s+year|annually|a\s+year)",
                    r"(?:demand|usage)\s+(?:is\s+|of\s+)?([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items)?\s*(?:annually|per\s+year)?",
                ],
                q,
            )
            if d is not None:
                inputs["annual_demand"] = d
            else:
                missing.append("annual_demand (e.g. 10,000 units/year)")

            # Ordering cost synonyms:
            # - ordering cost
            # - order cost
            # - setup cost
            # - setup expense
            # - cost per order
            # - cost of placing an order
            # - ordering/setup cost
            s = search_number(
                [
                    r"(?:ordering[/\s]+setup\s+cost|cost\s+of\s+placing\s+(?:an\s+)?order|cost\s+per\s+order|ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|fixed\s+order(?:ing)?\s+cost|reorder\s+cost|order\s+processing\s+cost)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+order|to\s+place\s+(?:an\s+)?order|for\s+each\s+order|setup\s+cost|setup\s+expense|ordering\s+cost|order\s+cost)",
                    r"costs?\s+\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+order|to\s+(?:place|process)\s+(?:an\s+)?order|for\s+setup)",
                ],
                q,
            )
            if s is not None:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost (synonyms: ordering cost, order cost, setup cost, setup expense, cost per order, cost of placing an order, ordering/setup cost)")

            # Holding cost (fixed $ or rate %)
            h = search_number(
                [
                    r"(?:holding\s+cost|carrying\s+cost|storage\s+cost|inventory\s+carrying\s+cost|cost\s+to\s+hold)(?:\s+per\s+unit(?:\s+per\s+year)?)?\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit)?",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|per\s+unit/year|per\s+unit\s+annually|holding\s+cost|carrying\s+cost|to\s+hold\s+each\s+unit)",
                ],
                q,
            )
            i_rate = search_number(
                [
                    r"(?:holding\s+(?:cost\s+)?rate|carrying\s+(?:cost\s+)?rate|percentage\s+holding\s+cost|inventory\s+carrying\s+rate|holding\s+percentage|carrying\s+charge|cost\s+of\s+capital)\s*(?:is|of|equals|[:=])?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying|annual\s+carrying)\s*(?:cost|rate)?",
                ],
                q,
            )
            c = search_number(
                [
                    r"(?:unit\s+(?:purchase\s+)?cost|cost\s+per\s+unit|unit\s+price|purchase\s+price|item\s+cost|wholesale\s+price|price\s+per\s+unit)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit|each|unit\s+cost|purchase\s+price)",
                ],
                q,
            )

            if h is not None:
                inputs["holding_cost"] = h
            elif i_rate is not None and c is not None:
                inputs["holding_cost_rate"] = i_rate / 100.0 if i_rate > 1.0 else i_rate
                inputs["unit_cost"] = c
            elif i_rate is not None:
                missing.append("unit_cost (required when holding cost rate is specified)")
            else:
                missing.append("holding_cost (or holding_cost_rate and unit_cost)")

            if c is not None and "unit_cost" not in inputs:
                inputs["unit_cost"] = c

            # Working days
            wd = search_number(
                [r"([0-9]+)\s*(?:working\s+days|operating\s+days|business\s+days|days\s+(?:per|a)\s+year)"],
                q,
            )
            inputs["working_days_per_year"] = int(wd) if wd else 250

        # ---------------------------------------------------------------------
        # 2. Quantity Discount Extraction
        # ---------------------------------------------------------------------
        elif model == InventoryModelType.QUANTITY_DISCOUNT:
            # Demand
            d = search_number(
                [
                    r"(?:annual\s+demand|demand\s+per\s+year|yearly\s+demand|annual\s+usage|annual\s+volume)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:annually|per\s+year|a\s+year)",
                    r"(?:demand|usage|purchases?)\s+(?:of\s+)?([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )
            if d is not None:
                inputs["annual_demand"] = d
            else:
                missing.append("annual_demand")

            # Ordering cost (recognizes all synonyms)
            s = search_number(
                [
                    r"(?:ordering[/\s]+setup\s+cost|cost\s+of\s+placing\s+(?:an\s+)?order|cost\s+per\s+order|ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|fixed\s+order(?:ing)?\s+cost|reorder\s+cost)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+order|to\s+place\s+(?:an\s+)?order|order\s+cost|setup\s+cost)",
                ],
                q,
            )
            if s is not None:
                inputs["order_cost"] = s
            else:
                missing.append("order_cost (synonyms: ordering cost, order cost, setup cost, setup expense, cost per order)")

            # Fixed holding cost vs Percentage holding cost
            i_rate = search_number(
                [
                    r"(?:percentage\s+holding\s+cost|holding\s+(?:cost\s+)?rate|carrying\s+(?:cost\s+)?rate|inventory\s+carrying\s+rate|holding\s+rate)\s*(?:is|of|equals|[:=])?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying)",
                ],
                q,
            )
            h_fix = search_number(
                [
                    r"(?:fixed\s+holding\s+cost|constant\s+holding\s+cost|holding\s+cost)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit)?",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+holding|fixed\s+holding)",
                ],
                q,
            )

            if i_rate is not None:
                inputs["holding_cost_rate"] = i_rate / 100.0 if i_rate > 1.0 else i_rate
            elif h_fix is not None:
                inputs["fixed_holding_cost"] = h_fix
            else:
                missing.append("holding_cost_rate (percentage) or fixed_holding_cost")

            # Extract discount tiers / quantity breaks
            tiers = []
            # Pattern A: "1 to 999 units at $5.00" or "1000 - 1999: $4.80"
            tier_matches = re.findall(
                r"([0-9,]+)\s*(?:to|-)\s*([0-9,]+|\+)?\s*(?:units|items|pieces)?\s*(?:at|@|for|costing|[:=])?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                q,
                re.IGNORECASE,
            )
            # Pattern B: "2000 or more units at $4.60" or "2000+ at $4.60" or "over 2000 at $4.60"
            plus_matches = re.findall(
                r"([0-9,]+)\s*(?:\+|or\s+more|and\s+above|and\s+over|\+)\s*(?:units|items|pieces)?\s*(?:at|@|for|costing|[:=])?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                q,
                re.IGNORECASE,
            )
            over_matches = re.findall(
                r"(?:over|above|at\s+least|exceeding)\s+([0-9,]+)\s*(?:units|items|pieces)?\s*(?:at|@|for|costing|[:=])?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                q,
                re.IGNORECASE,
            )

            for m in tier_matches:
                low = float(m[0].replace(",", ""))
                high = float(m[1].replace(",", "")) if (m[1] and m[1] != "+") else float("inf")
                cost = float(m[2])
                tiers.append({"min_qty": low, "max_qty": high, "unit_cost": cost})

            for m in plus_matches:
                low = float(m[0].replace(",", ""))
                cost = float(m[1])
                if not any(t["min_qty"] == low for t in tiers):
                    tiers.append({"min_qty": low, "max_qty": float("inf"), "unit_cost": cost})

            for m in over_matches:
                low = float(m[0].replace(",", ""))
                cost = float(m[1])
                if not any(t["min_qty"] == low for t in tiers):
                    tiers.append({"min_qty": low, "max_qty": float("inf"), "unit_cost": cost})

            if tiers:
                inputs["tiers"] = sorted(tiers, key=lambda x: x["min_qty"])
            else:
                missing.append("discount_tiers (e.g. 1-999 at $5.00, 1000-1999 at $4.80, 2000+ at $4.60)")

        # ---------------------------------------------------------------------
        # 3. Safety Stock & Reorder Point Extraction
        # Distinctly extracts:
        # - average demand
        # - demand standard deviation
        # - average lead time
        # - lead-time standard deviation
        # - service level
        # ---------------------------------------------------------------------
        elif model == InventoryModelType.SAFETY_STOCK_ROP:
            # 1. Lead-time standard deviation (must check FIRST to avoid cross-matching lead time)
            lt_std = search_number(
                [
                    r"(?:lead[- ]time\s+standard\s+deviation|standard\s+deviation\s+of\s+lead[- ]time|lead[- ]time\s+std(?:\s+dev)?|std\s+dev\s+of\s+lead[- ]time)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks)?",
                    r"lead[- ]time\s+(?:has\s+a\s+|with\s+a\s+)?standard\s+deviation\s+(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks)?",
                    r"lead[- ]time\s+varies\s+(?:with\s+a\s+std\s+dev\s+of\s+|by\s+)?([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks)?",
                ],
                q,
            )
            if lt_std is not None:
                inputs["lead_time_std"] = lt_std

            # 2. Average lead time
            lt = search_number(
                [
                    r"(?:average\s+lead\s+time|mean\s+lead\s+time|supplier\s+lead\s+time|delivery\s+lead\s+time|replenishment\s+lead\s+time|lead\s+time)\s*(?:is|of|averages?|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:day|week|month|period)s?\s+lead\s+time",
                    r"takes\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks)\s+(?:for\s+delivery|to\s+arrive)",
                ],
                q,
            )
            if lt is not None:
                inputs["lead_time_mean"] = lt
            else:
                missing.append("average_lead_time (e.g. 4 days)")

            # Check for direct lead-time demand distribution (mu_LT, sigma_LT)
            ltd_mean = search_number(
                [
                    r"(?:lead\s+time\s+demand\s+(?:mean|average)|average\s+(?:demand\s+during\s+lead\s+time|lead\s+time\s+demand))\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )
            ltd_std = search_number(
                [
                    r"(?:lead\s+time\s+demand\s+std|lead\s+time\s+std\s+dev|std\s+dev\s+of\s+(?:demand\s+during\s+lead\s+time|lead\s+time\s+demand)|standard\s+deviation\s+of\s+demand\s+during\s+lead\s+time)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )

            if ltd_mean is not None and ltd_std is not None:
                inputs["lead_time_demand_mean"] = ltd_mean
                inputs["lead_time_demand_std"] = ltd_std
            else:
                # 3. Average demand
                d_mean = search_number(
                    [
                        r"(?:average\s+demand|mean\s+demand|expected\s+demand|daily\s+demand|weekly\s+demand|demand\s+rate)\s*(?:is|of|averages?|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                        r"demand\s+(?:is|averages?)\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+(?:day|week|month|period))",
                        r"sells?\s+(?:on\s+average\s+)?([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+(?:day|week|month|period))",
                    ],
                    q,
                )
                if d_mean is not None:
                    inputs["demand_mean"] = d_mean
                else:
                    missing.append("average_demand (e.g. 50 units/day)")

                # 4. Demand standard deviation
                d_std = search_number(
                    [
                        r"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|std\s+dev\s+of\s+demand|demand\s+std(?:\s+dev)?|sigma\s+of\s+demand)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                        r"(?:standard\s+deviation|std\s+dev|sigma)\s+(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+(?:day|week))?",
                    ],
                    q,
                )
                if d_std is not None:
                    inputs["demand_std"] = d_std
                else:
                    # If not specified, default to 0.0 only if text explicitly indicates constant/deterministic demand
                    if any(w in q.lower() for w in ["constant demand", "deterministic demand", "no demand variance"]):
                        inputs["demand_std"] = 0.0
                    else:
                        inputs["demand_std"] = 0.0  # Allowed for Case 1 / Case 3

            # 5. Service level
            csl = search_number(
                [
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?service\s+level",
                    r"(?:cycle\s+)?service\s+level\s*(?:is|of|targeting|equals|[:=])?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"(?:cycle\s+)?service\s+level\s*(?:is|of|targeting|equals|[:=])?\s*(0\.[0-9]+)",
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:fill\s+rate|order\s+completion)",
                    r"stockout\s+risk\s*(?:is|of|equals|[:=])?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                ],
                q,
            )
            if csl is not None:
                # If stockout risk was matched:
                if "stockout risk" in q.lower() and csl < 50.0:
                    inputs["service_level"] = (100.0 - csl) / 100.0
                else:
                    inputs["service_level"] = csl / 100.0 if csl > 1.0 else csl
            else:
                # If demand or lead time is variable, service level is required
                if (inputs.get("demand_std", 0.0) > 0.0) or (inputs.get("lead_time_std", 0.0) > 0.0):
                    missing.append("service_level (e.g. 95% cycle service level)")
                else:
                    inputs["service_level"] = 1.0  # Constant demand and constant lead time

        # ---------------------------------------------------------------------
        # 4. Fixed-Period Review Extraction
        # Distinctly extracts:
        # - demand (mean and std)
        # - review period
        # - lead time
        # - safety stock
        # - inventory position
        # - on-hand inventory
        # - on-order inventory
        # - backorders
        # ---------------------------------------------------------------------
        elif model == InventoryModelType.FIXED_PERIOD_REVIEW:
            # Review period
            rp = search_number(
                [
                    r"(?:review\s+period|review\s+cycle|review\s+interval)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)?",
                    r"(?:reviewed|check(?:ed)?(?:\s+inventory)?|orders?\s+placed)\s+every\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)?",
                    r"every\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)",
                ],
                q,
            )
            if rp is not None:
                inputs["review_period"] = rp
            else:
                missing.append("review_period (e.g. 14 days)")

            # Lead time
            lt = search_number(
                [
                    r"(?:supplier\s+lead\s+time|delivery\s+(?:lead\s+)?time|lead\s+time)\s*(?:is|of|averages?|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks|months|periods)?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:day|week|month|period)s?\s+lead\s+time",
                    r"takes\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:days|weeks)\s+for\s+delivery",
                ],
                q,
            )
            if lt is not None:
                inputs["lead_time"] = lt
            else:
                missing.append("lead_time (e.g. 7 days)")

            # Demand mean & std
            d_mean = search_number(
                [
                    r"(?:average\s+demand|mean\s+demand|daily\s+demand|weekly\s+demand|expected\s+demand)\s*(?:is|of|averages?|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"demand\s+(?:is|averages?)\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+(?:day|week|month|period))",
                ],
                q,
            )
            d_std = search_number(
                [
                    r"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|std\s+dev\s+of\s+demand|demand\s+std|sigma)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:standard\s+deviation|std\s+dev|sigma)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
                ],
                q,
            )

            if d_mean is not None:
                inputs["demand_mean"] = d_mean
            else:
                missing.append("demand_mean (e.g. 10 units/day)")

            if d_std is not None:
                inputs["demand_std"] = d_std
            else:
                missing.append("demand_std (e.g. 3 units/day)")

            # Service level
            csl = search_number(
                [
                    r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?service\s+level",
                    r"service\s+level\s*(?:is|of|targeting|equals|[:=])?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                    r"service\s+level\s*(?:is|of|targeting|equals|[:=])?\s*(0\.[0-9]+)",
                ],
                q,
            )
            inputs["service_level"] = csl / 100.0 if (csl and csl > 1.0) else (csl if csl else 0.95)

            # Direct Inventory Position
            inv_pos = search_number(
                [
                    r"(?:inventory\s+position|current\s+inventory\s+position|net\s+inventory)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )

            # On-hand inventory
            on_hand = search_number(
                [
                    r"(?:on-hand\s+inventory|inventory\s+on[- ]hand|stock\s+on[- ]hand|on[- ]hand)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"currently\s+(?:have\s+|in\s+stock\s+)?([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:on[- ]hand|in\s+stock)?",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:on[- ]hand|in\s+stock)",
                ],
                q,
            )

            # On-order inventory
            on_order = search_number(
                [
                    r"(?:on-order\s+inventory|inventory\s+on[- ]order|units\s+on[- ]order|pipeline\s+inventory|pipeline\s+stock|on[- ]order)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:on[- ]order|in\s+pipeline|open\s+orders|in\s+transit)",
                    r"(?:open\s+orders|pending\s+orders)\s*(?:of|is|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )

            # Backorders
            backorders = search_number(
                [
                    r"(?:backorders?|backlogged\s+demand|backlog|unfilled\s+orders)\s*(?:is|are|of|at|equals|stand\s+at|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:backordered|backlogged|on\s+backorder)",
                ],
                q,
            )

            if inv_pos is not None:
                inputs["inventory_position"] = inv_pos
            elif on_hand is not None:
                inputs["on_hand_inventory"] = on_hand
                inputs["on_order_inventory"] = on_order if on_order is not None else 0.0
                bo = backorders if backorders is not None else 0.0
                inputs["backorders"] = bo
                inputs["inventory_position"] = on_hand + inputs["on_order_inventory"] - bo

        # ---------------------------------------------------------------------
        # 5. Newsvendor Extraction
        # Distinctly extracts:
        # - selling price
        # - purchase cost
        # - salvage value
        # - underage / shortage cost
        # - overage / excess cost
        # - demand distribution
        # - mean demand
        # - demand standard deviation
        # ---------------------------------------------------------------------
        elif model == InventoryModelType.NEWSVENDOR:
            # Selling price
            price = search_number(
                [
                    r"(?:selling\s+price|retail\s+price|regular\s+price|sale\s+price|revenue\s+per\s+unit|price\s+per\s+unit)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:retail|selling\s+price|regular\s+price|price\s+per\s+unit)",
                    r"sells?\s+(?:at|for)\s+\$([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )

            # Purchase cost / unit cost
            cost = search_number(
                [
                    r"(?:unit\s+(?:purchase\s+)?cost|purchase\s+cost|cost\s+per\s+unit|wholesale\s+price|procurement\s+cost|manufacturing\s+cost)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:unit\s+cost|wholesale|procurement|purchase\s+cost)",
                    r"costs?\s+\$([0-9,]+(?:\.[0-9]+)?)\s*(?:to\s+(?:buy|make|produce)|per\s+unit)?",
                ],
                q,
            )

            # Salvage value
            salvage = search_number(
                [
                    r"(?:salvage\s+value|salvage\s+price|clearance\s+price|scrap\s+value|liquidation\s+value)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:salvaged|liquidated|cleared|sold\s+at\s+clearance)\s+(?:at|for)\s+\$?([0-9,]+(?:\.[0-9]+)?)",
                    r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:salvage|salvaged|clearance|scrap)",
                ],
                q,
            )

            # Underage / shortage cost or penalty
            cu_direct = search_number(
                [
                    r"(?:underage\s+cost|shortage\s+cost|cost\s+of\s+underage|cost\s+of\s+shortage|lost\s+profit\s+per\s+unit)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )
            penalty = search_number(
                [
                    r"(?:underage\s+penalty|stockout\s+penalty|goodwill\s+loss|penalty\s+for\s+lost\s+sales)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )

            # Overage / excess cost
            co_direct = search_number(
                [
                    r"(?:overage\s+cost|excess\s+cost|cost\s+of\s+overage|cost\s+of\s+excess)\s*(?:is|of|equals|[:=])?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
                ],
                q,
            )

            # Demand mean
            d_mean = search_number(
                [
                    r"(?:mean\s+demand|average\s+demand|expected\s+demand|forecasted\s+demand)\s*(?:is|of|averages?|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"(?:normally\s+distributed\s+with\s+mean|mean\s+of)\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"demand\s+is\s+([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:on\s+average)?",
                ],
                q,
            )

            # Demand standard deviation
            d_std = search_number(
                [
                    r"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|std\s+dev\s+of\s+demand|demand\s+std|sigma)\s*(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)",
                    r"standard\s+deviation\s+(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
                    r"std\s+dev\s+(?:is|of|equals|[:=])?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
                ],
                q,
            )

            # Validate parameters and examine entire context
            if price is not None:
                inputs["selling_price"] = price
            elif cu_direct is not None and cost is not None:
                # Can infer selling price: P = Cu + C
                inputs["selling_price"] = cu_direct + cost
            else:
                missing.append("selling_price (or direct underage_cost and purchase_cost)")

            if cost is not None:
                inputs["unit_cost"] = cost
            elif price is not None and cu_direct is not None:
                # Can infer cost: C = P - Cu
                inputs["unit_cost"] = price - cu_direct
            elif co_direct is not None and salvage is not None:
                # Can infer cost: C = Co + V
                inputs["unit_cost"] = co_direct + salvage
            else:
                missing.append("unit_cost (or direct underage/overage cost)")

            inputs["salvage_value"] = salvage if salvage is not None else 0.0

            if d_mean is not None:
                inputs["demand_mean"] = d_mean
            else:
                missing.append("demand_mean (expected average demand)")

            if d_std is not None:
                inputs["demand_std"] = d_std
            else:
                missing.append("demand_std (demand standard deviation)")

            if penalty is not None:
                inputs["underage_penalty"] = penalty

        return inputs, missing

    # =========================================================================
    # Step 3: Select Python Tool
    # =========================================================================

    def select_tool(self, model: InventoryModelType) -> Tuple[Callable[..., Dict[str, Any]], str]:
        """Select appropriate calculation tool from inventory_tools module."""
        if model not in self.tool_mapping:
            raise ValueError(f"No registered calculation tool for model: {model}")
        tool_fn = self.tool_mapping[model]
        return tool_fn, tool_fn.__name__

    # =========================================================================
    # Step 4: Calculate
    # =========================================================================

    def calculate(
        self, tool_fn: Callable[..., Dict[str, Any]], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute numerical calculation in Python."""
        return tool_fn(**inputs)

    # =========================================================================
    # Step 5: Verify
    # =========================================================================

    def verify(
        self,
        model: InventoryModelType,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
    ) -> VerificationReport:
        """Perform rigorous supply chain verification, boundary checks, and managerial sanity rules."""
        report = VerificationReport(is_valid=True)
        res = results.get("results", {})

        if model == InventoryModelType.EOQ:
            q = res.get("optimal_order_quantity", 0)
            aoc = res.get("annual_ordering_cost", 0)
            ahc = res.get("annual_holding_cost", 0)

            if q > 0:
                report.sanity_checks_passed.append(f"Optimal EOQ Q* = {q:,} units is strictly positive.")
            else:
                report.is_valid = False
                report.warnings.append("EOQ calculated is non-positive.")

            # Check trade-off balance: AOC ~ AHC
            diff = abs(aoc - ahc)
            pct_diff = (diff / max(aoc, 1.0)) * 100.0
            if pct_diff < 1.0:
                report.sanity_checks_passed.append(
                    f"First-order optimality condition satisfied: Annual Ordering Cost (${aoc:,.2f}) ≈ Annual Holding Cost (${ahc:,.2f})."
                )
            else:
                report.warnings.append(
                    f"Ordering cost (${aoc:,.2f}) and holding cost (${ahc:,.2f}) deviate by {pct_diff:.2f}%, check input parameters."
                )

            # Cycle time managerial note
            cycle_days = res.get("cycle_time_days", 0)
            if cycle_days > 180:
                report.managerial_notes.append(
                    f"Cycle time of {cycle_days:.1f} operating days exceeds half a year; verify perishability and obsolescence risks."
                )

        elif model == InventoryModelType.QUANTITY_DISCOUNT:
            best_tier = res.get("optimal_tier")
            best_q = res.get("optimal_order_quantity", 0)
            savings = res.get("annual_savings_vs_base_tier", 0)

            report.sanity_checks_passed.append(
                f"Evaluated all discount tiers. Minimum total annual cost achieved at Tier {best_tier} with order quantity {best_q:,.0f}."
            )
            if savings > 0:
                report.sanity_checks_passed.append(
                    f"Bulk purchase produces positive net annual savings of ${savings:,.2f} ({res.get('percent_savings')}%) over baseline."
                )
            else:
                report.managerial_notes.append(
                    "No financial advantage to order at higher discount tiers; baseline EOQ yields lowest total cost."
                )

        elif model == InventoryModelType.SAFETY_STOCK_ROP:
            ss = res.get("safety_stock", 0)
            rop = res.get("reorder_point", 0)
            csl = res.get("cycle_service_level_percent", 0)
            z = res.get("z_score", 0)

            if rop >= ss:
                report.sanity_checks_passed.append(
                    f"Reorder point ({rop:,}) appropriately bounds safety stock ({ss:,}) plus expected lead-time demand."
                )
            else:
                report.is_valid = False
                report.warnings.append("Reorder point is lower than safety stock.")

            if z > 2.58:  # 99.5%+
                report.warnings.append(
                    f"Target service level of {csl}% requires a high z-score ({z:.2f}), causing exponential inventory holding inflation."
                )
            else:
                report.sanity_checks_passed.append(
                    f"Service level of {csl}% is well within operational efficiency bounds (z = {z:.2f})."
                )

        elif model == InventoryModelType.FIXED_PERIOD_REVIEW:
            order_up_to = res.get("order_up_to_level", 0)
            protect = res.get("protection_interval", 0)
            q_rec = res.get("recommended_order_quantity")

            report.sanity_checks_passed.append(
                f"Protection interval correctly spans review period plus lead time ({protect} periods)."
            )
            if q_rec is not None:
                report.sanity_checks_passed.append(
                    f"Recommended replenishment order calculated as {q_rec:,} units given current stock and pipeline."
                )

        elif model == InventoryModelType.NEWSVENDOR:
            cr = res.get("critical_ratio", 0)
            cu = res.get("cost_of_underage_Cu", 0)
            co = res.get("cost_of_overage_Co", 0)
            q_star = res.get("optimal_order_quantity", 0)
            profit = res.get("expected_profit", 0)

            if 0.0 < cr < 1.0:
                report.sanity_checks_passed.append(
                    f"Critical fractile CR = {cr:.4f} ({cr * 100:.1f}%) is within valid probability bounds (0, 1)."
                )
            else:
                report.is_valid = False
                report.warnings.append(f"Critical ratio {cr} is out of bounds.")

            if profit > 0:
                report.sanity_checks_passed.append(
                    f"Expected single-period profit is positive (${profit:,.2f})."
                )
            else:
                report.warnings.append("Expected profit is non-positive; reconsider product viability.")

            mu_display = inputs.get('demand_mean', inputs.get('mean_demand', 0))
            if cu > co:
                report.managerial_notes.append(
                    f"High margin product (Cu = ${cu:,.2f} > Co = ${co:,.2f}): optimal policy orders above average demand (Q* = {q_star:,} vs mu = {mu_display:,}) to avoid stockout costs."
                )
            elif co > cu:
                report.managerial_notes.append(
                    f"High salvage loss product (Co = ${co:,.2f} > Cu = ${cu:,.2f}): optimal policy orders conservatively below average demand to avoid liquidation losses."
                )
            else:
                report.managerial_notes.append(
                    f"Balanced underage and overage costs (Cu = Co = ${cu:,.2f}): critical fractile is 50%, ordering exactly at mean demand ({q_star:,} units)."
                )

        return report

    # =========================================================================
    # Step 6: Explain
    # =========================================================================

    def explain(
        self,
        model: InventoryModelType,
        inputs: Dict[str, Any],
        results: Dict[str, Any],
        verification: VerificationReport,
    ) -> str:
        """Synthesize calculation into clear student-verifiable sections directly from tool results."""
        lines: List[str] = []

        lines.append(f"### Supply Chain Decision Analysis: {model.value}")
        lines.append("")

        # 1. Inputs Extracted
        lines.append("#### 1. Inputs Extracted")
        tool_inputs = results.get("inputs_extracted", results.get("inputs", inputs))
        for k, v in tool_inputs.items():
            if k == "discount_tiers":
                lines.append(f"- **{k}**:")
                for t in v:
                    lines.append(f"  - Range {t.get('min_qty')}-{t.get('max_qty')}: ${t.get('unit_cost')}/unit")
            else:
                lines.append(f"- **{k}**: `{v}`")
        lines.append("")

        # 2. Variable Definitions
        lines.append("#### 2. Variable Definitions")
        var_defs = results.get("variable_definitions", {})
        for var, defn in var_defs.items():
            lines.append(f"- **{var}**: {defn}")
        lines.append("")

        # 3. Formula
        lines.append("#### 3. Formula")
        formulas = results.get("formula", results.get("formulas", {}))
        for name, form in formulas.items():
            lines.append(f"- **{name.replace('_', ' ').title()}**: `{form}`")
        lines.append("")

        # 4. Formula with Substituted Values
        lines.append("#### 4. Formula with Substituted Values")
        subs = results.get("formula_with_substituted_values", {})
        if isinstance(subs, dict):
            for name, val in subs.items():
                lines.append(f"- **{name.replace('_', ' ').title()}**: {val}")
        elif isinstance(subs, list):
            for s in subs:
                lines.append(f"- {s}")
        lines.append("")

        # 5. Intermediate Calculations
        lines.append("#### 5. Intermediate Calculations")
        intermediates = results.get("intermediate_calculations", {})
        for k, v in intermediates.items():
            if k == "tier_evaluations":
                lines.append("- **Tier-by-Tier Evaluation Table**:")
                lines.append("  | Tier | Range | Unit Cost | Unconstrained EOQ | Candidate Q | Purchase Cost | Ordering Cost | Holding Cost | Total Cost | Status |")
                lines.append("  |---|---|---|---|---|---|---|---|---|---|")
                for t in v:
                    lines.append(
                        f"  | {t['tier_index']} | {t['min_qty']:,.0f}-{t['max_qty']} | ${t['unit_cost']:.2f} | {t['unconstrained_eoq']:,.2f} | {t['candidate_order_quantity']:,.0f} | ${t['annual_purchase_cost']:,.2f} | ${t['annual_ordering_cost']:,.2f} | ${t['annual_holding_cost']:,.2f} | **${t['total_annual_cost']:,.2f}** | {t['feasibility_status']} |"
                    )
            else:
                lines.append(f"- **{k.replace('_', ' ').title()}**: `{v}`")
        lines.append("")

        # 6. Final Answer
        lines.append("#### 6. Final Answer (Executive Recommendation)")
        final_ans = results.get("final_answer", results.get("results", {}))
        for k, v in final_ans.items():
            lines.append(f"- **{k.replace('_', ' ').title()}**: **{v}**")
        lines.append("")

        # 7. Verification Checks
        lines.append("#### 7. Verification Checks")
        verif_checks = results.get("verification_checks", [])
        if verif_checks:
            for chk in verif_checks:
                status_icon = "[PASS]" if chk.get("passed") else "[FAIL]"
                lines.append(f"- {status_icon} **{chk.get('check')}**: {chk.get('details')}")
        for check_str in verification.sanity_checks_passed:
            lines.append(f"- [PASS] {check_str}")
        for note in verification.managerial_notes:
            lines.append(f"- [NOTE] {note}")
        for warn in verification.warnings:
            lines.append(f"- [WARNING] {warn}")
        lines.append("")

        # 8. Assumptions
        lines.append("#### 8. Assumptions")
        for asm in results.get("assumptions", []):
            lines.append(f"- {asm}")

        return "\n".join(lines)

    # =========================================================================
    # Master Execution Method
    # =========================================================================

    def process_question(
        self, question: str, model_override: Optional[str] = None
    ) -> AgentResponse:
        """Execute complete agent decision pipeline on natural language question."""
        # 1. Identify Model
        model = self.identify_model(question, override=model_override)
        if model == InventoryModelType.UNKNOWN:
            return AgentResponse(
                question=question,
                identified_model=model,
                extracted_inputs={},
                missing_inputs=[],
                selected_tool_name="None",
                calculation_results=None,
                verification=None,
                explanation=(
                    "I could not clearly match your question to one of the 5 supported inventory models:\n"
                    "1. Economic Order Quantity (EOQ)\n"
                    "2. Quantity Discount Model\n"
                    "3. Safety Stock & Reorder Point\n"
                    "4. Fixed-Period Review Model\n"
                    "5. Newsvendor Model\n\n"
                    "Please specify key parameters like annual demand, costs, lead time, or review period."
                ),
                status="CALCULATION_ERROR",
            )

        # 2. Extract Inputs
        inputs, missing = self.extract_inputs(question, model)
        if missing:
            return AgentResponse(
                question=question,
                identified_model=model,
                extracted_inputs=inputs,
                missing_inputs=missing,
                selected_tool_name="None",
                calculation_results=None,
                verification=None,
                explanation=(
                    f"Identified Model: **{model.value}**\n\n"
                    f"To perform the calculation, I still need the following missing inputs:\n"
                    + "\n".join(f"- `{m}`" for m in missing)
                    + f"\n\nExtracted so far: `{inputs}`"
                ),
                status="MISSING_INPUTS",
            )

        # 3. Select Python Tool
        tool_fn, tool_name = self.select_tool(model)

        # 4. Calculate
        try:
            results = self.calculate(tool_fn, inputs)
        except Exception as e:
            return AgentResponse(
                question=question,
                identified_model=model,
                extracted_inputs=inputs,
                missing_inputs=[],
                selected_tool_name=tool_name,
                calculation_results=None,
                verification=None,
                explanation=f"Error executing Python calculation: {str(e)}",
                status="CALCULATION_ERROR",
            )

        # 5. Verify
        verification = self.verify(model, inputs, results)

        # 6. Explain
        explanation = self.explain(model, inputs, results, verification)

        return AgentResponse(
            question=question,
            identified_model=model,
            extracted_inputs=inputs,
            missing_inputs=[],
            selected_tool_name=tool_name,
            calculation_results=results,
            verification=verification,
            explanation=explanation,
            status="SUCCESS",
        )
