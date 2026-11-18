"""Supply Chain Analytics Inventory Decision Agent Application.

Dual-mode interface:
1. Interactive CLI Mode: Run `python app.py` for rich terminal analysis with presets and freeform prompts.
2. Streamlit Web Mode: Run `streamlit run app.py` for an interactive browser dashboard.
"""

from __future__ import annotations

import sys
from typing import Optional

from inventory_agent import InventoryDecisionAgent, InventoryModelType

SAMPLE_QUESTIONS = [
    {
        "title": "1. Economic Order Quantity (EOQ)",
        "query": (
            "Our warehouse has an annual demand of 10,000 units. "
            "The order cost is $50 per order, and our holding cost is $2 per unit per year. "
            "What is our optimal order quantity, order frequency, and total inventory cost?"
        ),
    },
    {
        "title": "2. Quantity Discount Model",
        "query": (
            "We have an annual demand of 10,000 units and order cost is $49. "
            "Holding rate is 20%. The supplier offers discounts: "
            "1 to 999 units at $5.00, 1000 to 1999 units at $4.80, and 2000 or more units at $4.60. "
            "Which price tier should we choose?"
        ),
    },
    {
        "title": "3. Safety Stock & Reorder Point (Continuous Review)",
        "query": (
            "Average daily demand is 50 units with a standard deviation of 10 units. "
            "Supplier lead time is 4 days. What should our safety stock and reorder point be "
            "for a 95% cycle service level?"
        ),
    },
    {
        "title": "4. Fixed-Period Review Model (Periodic Review)",
        "query": (
            "We check inventory every 14 days. Delivery lead time is 7 days. "
            "Average demand is 10 units per day with std dev of 3 units. "
            "We target a 95% service level. Currently we have 40 units on hand and 20 on order. "
            "How many units should we order?"
        ),
    },
    {
        "title": "5. Newsvendor Model (Single-Period Perishable/Seasonal)",
        "query": (
            "We are ordering seasonal ski jackets. Retail price is $100, unit purchase cost is $60, "
            "and leftover jackets can be salvaged at $20. Demand is normally distributed with "
            "mean 1000 and standard deviation 200. How many units should we order?"
        ),
    },
]


# =============================================================================
# CLI Mode (Rich Console)
# =============================================================================

def run_cli():
    try:
        from rich import box
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print("Please install rich or dependencies: pip install -r requirements.txt")
        return

    console = Console()
    agent = InventoryDecisionAgent()

    console.print(
        Panel.fit(
            "[bold cyan]Supply Chain Analytics - Inventory Decision Agent[/bold cyan]\n"
            "[white]Pipeline: Question → Identify Model → Extract Inputs → Select Tool → Calculate → Verify → Explain[/white]\n"
            "[green]Supports: EOQ | Quantity Discount | Safety Stock & ROP | Fixed-Period Review | Newsvendor[/green]",
            border_style="cyan",
        )
    )

    def display_response(resp):
        console.print()
        pipeline_table = Table(title="[bold yellow]Agent Execution Pipeline[/bold yellow]", box=box.ROUNDED)
        pipeline_table.add_column("Stage", style="cyan", no_wrap=True)
        pipeline_table.add_column("Outcome", style="white")

        pipeline_table.add_row("1. Identified Model", f"[bold green]{resp.identified_model.value}[/bold green]")
        pipeline_table.add_row("2. Selected Tool", f"[blue]{resp.selected_tool_name}[/blue]")
        pipeline_table.add_row(
            "3. Extracted Inputs",
            ", ".join(f"{k}={v}" for k, v in resp.extracted_inputs.items()) if resp.extracted_inputs else "None",
        )
        status_color = "green" if resp.status == "SUCCESS" else "red"
        pipeline_table.add_row("4. Status", f"[{status_color}]{resp.status}[/{status_color}]")
        console.print(pipeline_table)

        if resp.calculation_results and "results" in resp.calculation_results:
            res_table = Table(title="[bold yellow]Numerical Results[/bold yellow]", box=box.SIMPLE_HEAVY)
            res_table.add_column("Metric", style="cyan")
            res_table.add_column("Value", style="magenta", justify="right")
            for k, v in resp.calculation_results["results"].items():
                res_table.add_row(k.replace("_", " ").title(), str(v))
            console.print(res_table)

        console.print()
        console.print(Panel(Markdown(resp.explanation), title="[bold green]Executive Report & Proof[/bold green]", border_style="green"))
        console.print()

    while True:
        console.print("\n[bold cyan]Select an option:[/bold cyan]")
        for idx, s in enumerate(SAMPLE_QUESTIONS, 1):
            console.print(f"  [yellow]{idx}[/yellow]. Run Preset: [white]{s['title']}[/white]")
        console.print("  [yellow]C[/yellow]. Enter a custom natural-language inventory question")
        console.print("  [yellow]Q[/yellow]. Quit")

        choice = console.input("\n[bold]Your choice (1-5, C, Q): [/bold]").strip()

        if choice.lower() in ("q", "quit", "exit"):
            console.print("[green]Exiting Inventory Agent. Goodbye![/green]")
            break

        if choice in ("1", "2", "3", "4", "5"):
            preset = SAMPLE_QUESTIONS[int(choice) - 1]
            console.print(Panel(preset["query"], title=f"[bold cyan]Input Question: {preset['title']}[/bold cyan]"))
            resp = agent.process_question(preset["query"])
            display_response(resp)
        elif choice.lower() == "c":
            user_q = console.input("[bold]Enter your inventory question:[/bold]\n> ").strip()
            if not user_q:
                continue
            resp = agent.process_question(user_q)
            display_response(resp)
        else:
            console.print("[red]Invalid choice. Please select 1-5, C, or Q.[/red]")


# =============================================================================
# Streamlit Mode (Browser UI)
# =============================================================================

def run_streamlit():
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
    )

    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.markdown(
        "**Architecture**: `Customer Question` ➔ `Identify Model` ➔ `Extract Inputs` ➔ "
        "`Select Python Tool` ➔ `Calculate` ➔ `Verify` ➔ `Explain`"
    )

    agent = InventoryDecisionAgent()

    # Sidebar: Preset Scenarios
    st.sidebar.header("🎯 Example Supply Chain Scenarios")
    selected_preset = st.sidebar.selectbox(
        "Load a benchmark query:",
        options=["(Custom Prompt)"] + [s["title"] for s in SAMPLE_QUESTIONS],
    )

    default_text = ""
    if selected_preset != "(Custom Prompt)":
        for s in SAMPLE_QUESTIONS:
            if s["title"] == selected_preset:
                default_text = s["query"]
                break

    # Model override option
    model_override = st.sidebar.selectbox(
        "Model Classifier Override (Optional):",
        options=["Auto-Detect", "EOQ", "Quantity Discount", "Safety Stock & ROP", "Fixed-Period Review", "Newsvendor"],
    )
    override_val = None if model_override == "Auto-Detect" else model_override

    # Sidebar: Data Upload Section (last section in sidebar)
    st.sidebar.markdown("---")
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    # Query Input
    user_query = st.text_area(
        "Enter your natural-language inventory question:",
        value=default_text,
        height=120,
        placeholder="e.g. Annual demand is 10,000 widgets, order cost is $50, holding cost is $2. What is our EOQ?",
    )

    if st.button("🚀 Run Decision Agent", type="primary"):
        if not user_query.strip():
            st.warning("Please enter or select a question first.")
            return

        with st.spinner("Executing decision pipeline..."):
            resp = agent.process_question(user_query, model_override=override_val)

        # Pipeline progress card
        st.subheader("1. Agent Pipeline Execution")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Identified Model", resp.identified_model.value)
        c2.metric("Selected Tool", resp.selected_tool_name)
        c3.metric("Status", resp.status)
        valid_status = "PASSED" if (resp.verification and resp.verification.is_valid) else "WARNING/N/A"
        c4.metric("Verification", valid_status)

        # Extracted Inputs
        with st.expander("🔍 Extracted Parameters", expanded=True):
            st.json(resp.extracted_inputs)

        # Numerical Results Table
        if resp.calculation_results and "results" in resp.calculation_results:
            st.subheader("2. Python Numerical Calculation Results")
            results_dict = resp.calculation_results["results"]
            cols = st.columns(min(len(results_dict), 4))
            for i, (k, v) in enumerate(list(results_dict.items())[:4]):
                cols[i].metric(label=k.replace("_", " ").title(), value=f"{v}")

            with st.expander("Full Numerical Data Payload"):
                st.write(results_dict)

        # Executive Explanation
        st.subheader("3. Executive Managerial Explanation & Verification")
        st.markdown(resp.explanation)

    # =========================================================================
    # Uploaded CSV Data Processing & Analysis
    # =========================================================================
    if uploaded_file is not None:
        st.markdown("---")
        try:
            import pandas as pd
            from inventory_tools import calculate_multicriteria_classification

            df = pd.read_csv(uploaded_file)

            # Check for Classification schema
            col_map = {c.strip().lower(): c for c in df.columns}
            has_sales = any(k in col_map for k in ["sales ($)", "sales", "revenue", "sales_usd", "total sales"])
            has_hits = any(k in col_map for k in ["hits", "hit", "picks", "transactions", "order lines"])
            is_classification_csv = has_sales and has_hits

            # Check for Units & Unit Cost analysis schema
            units_required = ["Product", "Category", "Units", "Unit Cost"]
            is_units_analysis_csv = all(col in df.columns for col in units_required)

            # If neither schema matches, show clear guidance
            if not is_classification_csv and not is_units_analysis_csv:
                st.error(
                    "⚠️ Uploaded CSV does not match a supported inventory schema.\n\n"
                    "- **For Multicriteria Inventory Classification**, expected columns: "
                    "`Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`\n"
                    "- **For Product Units & Cost Analysis**, expected columns: "
                    "`Product`, `Category`, `Units`, `Unit Cost`"
                )

            # -----------------------------------------------------------------
            # 1. Multicriteria Inventory Classification (A, B, C, D)
            # -----------------------------------------------------------------
            if is_classification_csv:
                st.header("🏷️ Multicriteria Inventory Classification (A, B, C, D)")
                st.markdown(
                    "Classifies catalog items using two critical operational dimensions: "
                    "**Sales ($)** (financial impact / revenue) and **Hits** (warehouse picking velocity / order frequency)."
                )

                # Classification Method Selector
                c_method_col1, c_method_col2 = st.columns([2, 1])
                with c_method_col1:
                    method_selection = st.radio(
                        "Classification Methodology:",
                        options=[
                            "Cross-Criteria Matrix (Sales ABC × Hits ABC)",
                            "Composite Percentile Score (Weighted 50% Sales / 50% Hits)",
                        ],
                        horizontal=True,
                    )
                chosen_method = "composite" if "Composite" in method_selection else "matrix"

                # Perform multicriteria classification
                class_res = calculate_multicriteria_classification(df, method=chosen_method)
                classified_df = class_res["classified_df"]
                breakdown = class_res["class_breakdown"]
                results_summary = class_res["results"]

                # Executive Metric Cards for A, B, C, D
                st.subheader("Classification Summary by Class Tier")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric(
                    "Class A (Vital Fast-Movers)",
                    f"{breakdown['A']['count']} items",
                    f"{breakdown['A']['percent_items']}% catalog | {breakdown['A']['percent_sales']}% sales",
                )
                col_b.metric(
                    "Class B (High Value / Moderate)",
                    f"{breakdown['B']['count']} items",
                    f"{breakdown['B']['percent_items']}% catalog | {breakdown['B']['percent_sales']}% sales",
                )
                col_c.metric(
                    "Class C (Routine Commodities)",
                    f"{breakdown['C']['count']} items",
                    f"{breakdown['C']['percent_items']}% catalog | {breakdown['C']['percent_sales']}% sales",
                )
                col_d.metric(
                    "Class D (Slow-Moving Tail)",
                    f"{breakdown['D']['count']} items",
                    f"{breakdown['D']['percent_items']}% catalog | {breakdown['D']['percent_sales']}% sales",
                )

                # Total catalog summary metrics
                tot_col1, tot_col2, tot_col3, tot_col4 = st.columns(4)
                tot_col1.metric("Total Catalog SKUs", f"{results_summary['total_items']:,}")
                tot_col2.metric("Total Catalog Sales", f"${results_summary['total_sales']:,.2f}")
                tot_col3.metric("Total Warehouse Hits", f"{results_summary['total_hits']:,}")
                tot_col4.metric("Applied Strategy", chosen_method.title())

                # Distribution Breakdown Table
                st.subheader("Multicriteria Breakdown & Control Strategy")
                breakdown_table_data = []
                for c_name in ["A", "B", "C", "D"]:
                    b = breakdown[c_name]
                    breakdown_table_data.append({
                        "Class": f"Class {c_name}",
                        "SKU Count": b["count"],
                        "SKU %": f"{b['percent_items']}%",
                        "Total Sales ($)": b["total_sales"],
                        "Sales %": f"{b['percent_sales']}%",
                        "Total Hits": b["total_hits"],
                        "Hits %": f"{b['percent_hits']}%",
                        "Recommended Management Strategy": b["strategy"],
                    })
                st.dataframe(
                    pd.DataFrame(breakdown_table_data),
                    use_container_width=True,
                    column_config={
                        "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%.2f"),
                        "Total Hits": st.column_config.NumberColumn("Total Hits", format="%d"),
                    },
                    hide_index=True,
                )

                # Enriched Data Table with Classification Results
                st.subheader("Classified Inventory Catalog")
                # Detect exact column names used
                sales_col_used = class_res["inputs_extracted"]["sales_column"]
                hits_col_used = class_res["inputs_extracted"]["hits_column"]

                col_config = {
                    sales_col_used: st.column_config.NumberColumn(sales_col_used, format="$%.2f"),
                    hits_col_used: st.column_config.NumberColumn(hits_col_used, format="%d"),
                    "Composite Score": st.column_config.NumberColumn("Composite Score", format="%.2f"),
                    "Classification": st.column_config.TextColumn("Classification (A/B/C/D)"),
                }
                st.dataframe(
                    classified_df,
                    use_container_width=True,
                    column_config=col_config,
                )

                # Technical & Verification Details Expander
                with st.expander("🔍 Verification Checks & Mathematical Formulations"):
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.markdown("**Formulas Applied:**")
                        for f in class_res["formula"]:
                            st.markdown(f"- `{f}`")
                    with v_col2:
                        st.markdown("**Deterministic Verification Checks:**")
                        for chk in class_res["verification_checks"]:
                            badge = "✅" if chk["status"] == "PASSED" else "❌"
                            st.markdown(f"- {badge} {chk['check']}: **{chk['status']}**")

            # -----------------------------------------------------------------
            # 2. Product Units & Cost Analysis (Existing Capability Maintained)
            # -----------------------------------------------------------------
            if is_units_analysis_csv:
                st.header("📊 Inventory Data Upload & Analysis")

                # Ensure numeric types for calculation columns
                df_units = df.copy()
                df_units["Units"] = pd.to_numeric(df_units["Units"], errors="coerce").fillna(0)
                df_units["Unit Cost"] = pd.to_numeric(df_units["Unit Cost"], errors="coerce").fillna(0.0)

                # 5. Create new calculated column: Inventory Value = Units × Unit Cost
                df_units["Inventory Value"] = df_units["Units"] * df_units["Unit Cost"]

                # 1. Total Units
                total_units = int(df_units["Units"].sum())

                # 2. Average Unit Cost
                avg_unit_cost = float(df_units["Unit Cost"].mean()) if len(df_units) > 0 else 0.0

                # 3. Product with highest Units
                if len(df_units) > 0:
                    max_unit_idx = df_units["Units"].idxmax()
                    highest_product = str(df_units.loc[max_unit_idx, "Product"])
                    highest_product_units = int(df_units.loc[max_unit_idx, "Units"])
                else:
                    highest_product = "N/A"
                    highest_product_units = 0

                # 4. Group data by Category and calculate total Units
                units_by_category = (
                    df_units.groupby("Category", as_index=False)["Units"]
                    .sum()
                    .sort_values(by="Units", ascending=False)
                )

                # 6. Total Inventory Value across all products
                total_inventory_value = float(df_units["Inventory Value"].sum())

                # Display Results: Summary Metrics Cards
                st.subheader("Inventory Summary Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Units", f"{total_units:,}")
                m2.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
                m3.metric("Product with Highest Units", f"{highest_product} ({highest_product_units:,})")
                m4.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")

                # Display Dataset Information: rows, columns, column names
                st.subheader("Dataset Information")
                st.write(f"- **Number of Rows**: {df_units.shape[0]:,}")
                st.write(f"- **Number of Columns**: {df_units.shape[1]:,}")
                st.write(f"- **Column Names**: {', '.join(df_units.columns.tolist())}")

                # Display Total Units by Category
                st.subheader("Total Units by Category")
                st.dataframe(
                    units_by_category,
                    use_container_width=True,
                    column_config={
                        "Category": "Category",
                        "Units": st.column_config.NumberColumn("Total Units", format="%d"),
                    },
                )

                # Display the uploaded dataframe including the new Inventory Value column with currency formatting
                st.subheader("Uploaded Data Table (with Calculated Inventory Value)")
                st.dataframe(
                    df_units,
                    use_container_width=True,
                    column_config={
                        "Product": "Product",
                        "Category": "Category",
                        "Units": st.column_config.NumberColumn("Units", format="%d"),
                        "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                        "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f"),
                    },
                )

        except Exception as err:
            st.error(f"Error processing CSV file: {err}")


# =============================================================================
# Entry Point
# =============================================================================

def is_running_under_streamlit() -> bool:
    """Detect if running via `streamlit run app.py`."""
    return "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv)


if __name__ == "__main__":
    # If explicitly run via streamlit or flag
    if "--cli" in sys.argv:
        run_cli()
    elif "--web" in sys.argv:
        run_streamlit()
    elif is_running_under_streamlit():
        run_streamlit()
    else:
        # Default: Launch rich interactive CLI mode
        run_cli()
