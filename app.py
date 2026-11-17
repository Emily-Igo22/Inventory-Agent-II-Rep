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
