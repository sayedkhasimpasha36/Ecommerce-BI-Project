"""
AI-Powered E-Commerce Business Intelligence & Decision Support Dashboard
Single-file Streamlit application
Dataset: global_ecommerce_sales.csv (2,000 rows × 15 columns)
"""

import os
import warnings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional colour palette
COLORS = {
    "primary": "#1F3A5F",
    "secondary": "#2E86AB",
    "accent": "#E84855",
    "success": "#3BB273",
    "warning": "#F4A261",
    "neutral": "#6C757D",
    "bg_card": "#F8F9FA",
    "border": "#DEE2E6",
}

CATEGORY_COLORS = {
    "Technology": "#2E86AB",
    "Furniture": "#3BB273",
    "Office Supplies": "#F4A261",
    "Clothing & Accessories": "#E84855",
}

REGION_COLORS = px.colors.qualitative.Set2

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Main layout ── */
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    h1 { color: #1F3A5F; font-size: 1.9rem; font-weight: 700; }
    h2 { color: #1F3A5F; font-size: 1.4rem; font-weight: 600; margin-top: 1.4rem; }
    h3 { color: #2E86AB; font-size: 1.1rem; font-weight: 600; }

    /* ── KPI card ── */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #DEE2E6;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #6C757D;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1F3A5F;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #6C757D;
        margin-top: 0.2rem;
    }
    .kpi-positive { color: #3BB273; }
    .kpi-negative { color: #E84855; }

    /* ── Insight box ── */
    .insight-box {
        background: #EAF4FB;
        border-left: 4px solid #2E86AB;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .insight-box h4 { color: #1F3A5F; margin: 0 0 0.3rem 0; font-size: 0.95rem; }
    .insight-box p  { color: #374151; margin: 0; font-size: 0.88rem; line-height: 1.5; }

    /* ── Risk / Opportunity cards ── */
    .risk-card {
        background: #FFF5F5;
        border-left: 4px solid #E84855;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .opp-card {
        background: #F0FFF4;
        border-left: 4px solid #3BB273;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .card-title { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.5rem; color: #1F3A5F; }
    .card-label { font-weight: 600; font-size: 0.82rem; color: #6C757D; text-transform: uppercase; }
    .card-body  { font-size: 0.87rem; color: #374151; margin-bottom: 0.4rem; line-height: 1.45; }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid #DEE2E6;
        margin: 1.5rem 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background-color: #1F3A5F; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label { color: #CBD5E1 !important; font-size: 0.8rem; }

    /* ── Page nav ── */
    .nav-selected { background: #2E86AB !important; color: white !important; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str = "global_ecommerce_sales.csv") -> pd.DataFrame:
    """Load, validate and enrich the dataset.  Original CSV is never modified."""
    if not os.path.exists(path):
        st.error(f"Dataset not found at '{path}'.  Place global_ecommerce_sales.csv in the same directory as app.py.")
        st.stop()

    df = pd.read_csv(path)

    # ── Column name normalisation ──────────────────────────────────────────
    df.columns = df.columns.str.strip()

    required_cols = [
        "Order_ID", "Order_Date", "Customer_Name", "Customer_Segment",
        "Country", "Region", "Product_Category", "Product_Name",
        "Quantity", "Unit_Price", "Discount_Percent", "Total_Sales",
        "Shipping_Cost", "Profit", "Payment_Method",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing expected columns: {missing_cols}")
        st.stop()

    # ── Date parsing ───────────────────────────────────────────────────────
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    invalid_dates = df["Order_Date"].isna().sum()
    if invalid_dates:
        st.warning(f"{invalid_dates} rows have unparseable Order_Date values and will be excluded.")
        df = df.dropna(subset=["Order_Date"])

    # ── Numeric coercion ───────────────────────────────────────────────────
    for col in ["Quantity", "Unit_Price", "Discount_Percent", "Total_Sales", "Shipping_Cost", "Profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Remove rows where core numeric values are all NaN ─────────────────
    df = df.dropna(subset=["Total_Sales", "Profit"])

    # ── Derived time columns ───────────────────────────────────────────────
    df["Year"]          = df["Order_Date"].dt.year
    df["Month"]         = df["Order_Date"].dt.month
    df["Month_Name"]    = df["Order_Date"].dt.strftime("%b")
    df["YearMonth"]     = df["Order_Date"].dt.to_period("M").astype(str)
    df["Quarter"]       = df["Order_Date"].dt.quarter.apply(lambda q: f"Q{q}")

    # ── Derived KPI columns ────────────────────────────────────────────────
    df["Profit_Margin"] = df.apply(
        lambda r: (r["Profit"] / r["Total_Sales"] * 100) if r["Total_Sales"] != 0 else 0,
        axis=1,
    )
    df["Is_Loss"] = df["Profit"] < 0

    return df


RAW_DF = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar(df: pd.DataFrame):
    """Render sidebar filters and return the filtered dataframe."""
    with st.sidebar:
        st.markdown("## 📊 Dashboard Filters")
        st.markdown("---")

        # Year
        all_years = sorted(df["Year"].unique().tolist())
        sel_years = st.multiselect("Year", options=all_years, default=all_years)

        # Region
        all_regions = sorted(df["Region"].unique().tolist())
        sel_regions = st.multiselect("Region", options=all_regions, default=all_regions)

        # Country (depends on region selection)
        country_pool = sorted(df[df["Region"].isin(sel_regions)]["Country"].unique().tolist()) if sel_regions else sorted(df["Country"].unique().tolist())
        sel_countries = st.multiselect("Country", options=country_pool, default=country_pool)

        # Category
        all_cats = sorted(df["Product_Category"].unique().tolist())
        sel_cats = st.multiselect("Product Category", options=all_cats, default=all_cats)

        # Segment
        all_segs = sorted(df["Customer_Segment"].unique().tolist())
        sel_segs = st.multiselect("Customer Segment", options=all_segs, default=all_segs)

        # Payment method
        all_pays = sorted(df["Payment_Method"].unique().tolist())
        sel_pays = st.multiselect("Payment Method", options=all_pays, default=all_pays)

        # Discount range
        disc_min = float(df["Discount_Percent"].min())
        disc_max = float(df["Discount_Percent"].max())
        sel_disc = st.slider(
            "Discount % Range",
            min_value=disc_min,
            max_value=disc_max,
            value=(disc_min, disc_max),
            step=1.0,
        )

        st.markdown("---")
        st.caption(f"Dataset: {len(df):,} total orders")

    # Apply filters
    mask = (
        df["Year"].isin(sel_years if sel_years else all_years) &
        df["Region"].isin(sel_regions if sel_regions else all_regions) &
        df["Country"].isin(sel_countries if sel_countries else country_pool) &
        df["Product_Category"].isin(sel_cats if sel_cats else all_cats) &
        df["Customer_Segment"].isin(sel_segs if sel_segs else all_segs) &
        df["Payment_Method"].isin(sel_pays if sel_pays else all_pays) &
        df["Discount_Percent"].between(sel_disc[0], sel_disc[1])
    )
    filtered = df[mask].copy()

    if filtered.empty:
        st.warning("No data matches the current filter selection.  Please broaden your filters.")
        st.stop()

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def fmt_currency(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:,.2f}"


def fmt_pct(val: float) -> str:
    return f"{val:.1f}%"


def kpi_card(label: str, value: str, sub: str = "", colour_class: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    val_class = f"kpi-value {colour_class}".strip()
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="{val_class}">{value}</div>
        {sub_html}
    </div>"""


def insight_box(title: str, body: str) -> str:
    return f"""
    <div class="insight-box">
        <h4>💡 {title}</h4>
        <p>{body}</p>
    </div>"""


def risk_card(title: str, fact: str, insight: str, risk: str, action: str) -> str:
    return f"""
    <div class="risk-card">
        <div class="card-title">🔴 {title}</div>
        <div class="card-label">FACT</div>
        <div class="card-body">{fact}</div>
        <div class="card-label">INSIGHT</div>
        <div class="card-body">{insight}</div>
        <div class="card-label">RISK</div>
        <div class="card-body">{risk}</div>
        <div class="card-label">RECOMMENDED ACTION</div>
        <div class="card-body">{action}</div>
    </div>"""


def opp_card(title: str, fact: str, insight: str, opportunity: str, action: str) -> str:
    return f"""
    <div class="opp-card">
        <div class="card-title">🟢 {title}</div>
        <div class="card-label">FACT</div>
        <div class="card-body">{fact}</div>
        <div class="card-label">INSIGHT</div>
        <div class="card-body">{insight}</div>
        <div class="card-label">OPPORTUNITY</div>
        <div class="card-body">{opportunity}</div>
        <div class="card-label">RECOMMENDED ACTION</div>
        <div class="card-body">{action}</div>
    </div>"""


def standard_layout(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=20, t=40, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial", size=12, color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DEE2E6")
    fig.update_yaxes(gridcolor="#F3F4F6", linecolor="#DEE2E6")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_executive_overview(df: pd.DataFrame):
    st.markdown("# 📊 Executive Overview")
    st.markdown("High-level performance summary across all selected filters.")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────
    total_sales     = df["Total_Sales"].sum()
    total_profit    = df["Profit"].sum()
    total_orders    = df["Order_ID"].nunique()
    total_qty       = df["Quantity"].sum()
    unique_customers = df["Customer_Name"].nunique()
    aov             = total_sales / total_orders if total_orders else 0
    profit_margin   = (total_profit / total_sales * 100) if total_sales else 0
    avg_discount    = df["Discount_Percent"].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(kpi_card("Total Sales", fmt_currency(total_sales)), unsafe_allow_html=True)
    with col2:
        colour = "kpi-positive" if total_profit >= 0 else "kpi-negative"
        st.markdown(kpi_card("Total Profit", fmt_currency(total_profit), colour_class=colour), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("Total Orders", f"{total_orders:,}"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi_card("Total Quantity Sold", f"{total_qty:,}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(kpi_card("Unique Customers", f"{unique_customers:,}"), unsafe_allow_html=True)
    with col6:
        st.markdown(kpi_card("Avg Order Value", fmt_currency(aov)), unsafe_allow_html=True)
    with col7:
        margin_colour = "kpi-positive" if profit_margin >= 15 else ("kpi-negative" if profit_margin < 0 else "")
        st.markdown(kpi_card("Profit Margin", fmt_pct(profit_margin), colour_class=margin_colour), unsafe_allow_html=True)
    with col8:
        st.markdown(kpi_card("Avg Discount", fmt_pct(avg_discount)), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Monthly Trends ───────────────────────────────────────────────────
    st.markdown("## Monthly Sales & Profit Trends")
    monthly = (
        df.groupby("YearMonth", sort=True)
        .agg(Sales=("Total_Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(x=monthly["YearMonth"], y=monthly["Sales"], name="Sales",
               marker_color=COLORS["secondary"], opacity=0.75),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(x=monthly["YearMonth"], y=monthly["Profit"], name="Profit",
                   mode="lines+markers", line=dict(color=COLORS["success"], width=2.5),
                   marker=dict(size=5)),
        secondary_y=True,
    )
    fig_trend.update_layout(
        height=360, margin=dict(l=30, r=30, t=40, b=60),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial", size=12, color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        title="Monthly Sales (bars) & Profit (line)",
    )
    fig_trend.update_xaxes(tickangle=45, showgrid=False)
    fig_trend.update_yaxes(title_text="Sales ($)", secondary_y=False, gridcolor="#F3F4F6")
    fig_trend.update_yaxes(title_text="Profit ($)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Region & Category ────────────────────────────────────────────────
    st.markdown("## Sales Distribution")
    col_r, col_c = st.columns(2)

    with col_r:
        reg_sales = (
            df.groupby("Region")["Total_Sales"].sum()
            .sort_values(ascending=True).reset_index()
        )
        fig_reg = px.bar(
            reg_sales, x="Total_Sales", y="Region", orientation="h",
            title="Sales by Region", color="Region",
            color_discrete_sequence=REGION_COLORS,
            labels={"Total_Sales": "Total Sales ($)", "Region": ""},
        )
        fig_reg = standard_layout(fig_reg)
        fig_reg.update_layout(showlegend=False)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_c:
        cat_sales = df.groupby("Product_Category")["Total_Sales"].sum().reset_index()
        fig_cat = px.pie(
            cat_sales, values="Total_Sales", names="Product_Category",
            title="Sales by Product Category",
            color="Product_Category",
            color_discrete_map=CATEGORY_COLORS,
            hole=0.4,
        )
        fig_cat = standard_layout(fig_cat)
        fig_cat.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── AI Business Analyst ──────────────────────────────────────────────
    st.markdown("## 🤖 AI Business Analyst — Executive Summary")
    st.caption("Rule-based insights computed from the filtered dataset. No values are fabricated.")

    # Best / worst category
    cat_summary = df.groupby("Product_Category").agg(
        Sales=("Total_Sales", "sum"), Profit=("Profit", "sum")
    ).reset_index()
    cat_summary["Margin"] = cat_summary["Profit"] / cat_summary["Sales"] * 100
    best_cat  = cat_summary.loc[cat_summary["Profit"].idxmax(), "Product_Category"]
    worst_cat = cat_summary.loc[cat_summary["Profit"].idxmin(), "Product_Category"]
    best_cat_margin  = cat_summary.loc[cat_summary["Product_Category"] == best_cat,  "Margin"].values[0]
    worst_cat_margin = cat_summary.loc[cat_summary["Product_Category"] == worst_cat, "Margin"].values[0]

    # Best region
    reg_summary = df.groupby("Region").agg(Sales=("Total_Sales", "sum"), Profit=("Profit", "sum")).reset_index()
    best_reg   = reg_summary.loc[reg_summary["Sales"].idxmax(), "Region"]
    best_reg_s = reg_summary.loc[reg_summary["Region"] == best_reg, "Sales"].values[0]

    # Loss rate
    loss_orders = df["Is_Loss"].sum()
    loss_pct    = loss_orders / len(df) * 100

    # Discount vs profit correlation bucket
    high_disc   = df[df["Discount_Percent"] >= 20]
    low_disc    = df[df["Discount_Percent"] < 20]
    avg_margin_high = high_disc["Profit_Margin"].mean() if not high_disc.empty else 0
    avg_margin_low  = low_disc["Profit_Margin"].mean()  if not low_disc.empty  else 0

    insights_html = ""
    insights_html += insight_box(
        "Revenue & Profitability Snapshot",
        f"The filtered dataset shows total sales of <b>{fmt_currency(total_sales)}</b> and total profit of "
        f"<b>{fmt_currency(total_profit)}</b>, yielding an overall profit margin of <b>{fmt_pct(profit_margin)}</b>. "
        f"Average order value stands at <b>{fmt_currency(aov)}</b> with a mean discount of <b>{fmt_pct(avg_discount)}</b>."
    )
    insights_html += insight_box(
        "Category Leadership",
        f"<b>{best_cat}</b> generates the highest absolute profit with a margin of "
        f"<b>{fmt_pct(best_cat_margin)}</b>. "
        f"<b>{worst_cat}</b> delivers the lowest profit (<b>{fmt_pct(worst_cat_margin)}</b> margin), "
        "indicating a category that warrants a pricing or discount policy review."
    )
    insights_html += insight_box(
        "Regional Performance",
        f"<b>{best_reg}</b> is the leading region by revenue, contributing "
        f"<b>{fmt_currency(best_reg_s)}</b> in sales "
        f"({fmt_pct(best_reg_s/total_sales*100 if total_sales else 0)} of total)."
    )
    insights_html += insight_box(
        "Loss-Making Orders",
        f"<b>{loss_orders:,} orders ({fmt_pct(loss_pct)})</b> in the filtered dataset report negative profit. "
        "These are preserved in all calculations as legitimate transactions."
    )
    if not high_disc.empty:
        insights_html += insight_box(
            "Discount Impact on Margin",
            f"Orders with discount ≥ 20 % show an average profit margin of <b>{fmt_pct(avg_margin_high)}</b>, "
            f"compared to <b>{fmt_pct(avg_margin_low)}</b> for orders below 20 % discount. "
            "This pattern suggests heavy discounting may be compressing margins."
        )

    st.markdown(insights_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — SALES & PRODUCT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_sales_product(df: pd.DataFrame):
    st.markdown("# 📦 Sales & Product Analysis")
    st.markdown("Deep-dive into category performance, top/bottom products, and discount dynamics.")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Category Sales & Profit ──────────────────────────────────────────
    st.markdown("## Category Performance")
    cat_perf = df.groupby("Product_Category").agg(
        Sales   =("Total_Sales", "sum"),
        Profit  =("Profit",      "sum"),
        Orders  =("Order_ID",    "nunique"),
        Quantity=("Quantity",    "sum"),
    ).reset_index()
    cat_perf["Margin"] = cat_perf["Profit"] / cat_perf["Sales"] * 100

    col1, col2 = st.columns(2)
    with col1:
        fig_cs = px.bar(
            cat_perf.sort_values("Sales", ascending=True),
            x="Sales", y="Product_Category", orientation="h",
            title="Sales by Category", color="Product_Category",
            color_discrete_map=CATEGORY_COLORS,
            labels={"Sales": "Total Sales ($)", "Product_Category": ""},
        )
        fig_cs = standard_layout(fig_cs)
        fig_cs.update_layout(showlegend=False)
        st.plotly_chart(fig_cs, use_container_width=True)

    with col2:
        fig_cp = px.bar(
            cat_perf.sort_values("Profit", ascending=True),
            x="Profit", y="Product_Category", orientation="h",
            title="Profit by Category", color="Product_Category",
            color_discrete_map=CATEGORY_COLORS,
            labels={"Profit": "Total Profit ($)", "Product_Category": ""},
        )
        fig_cp = standard_layout(fig_cp)
        fig_cp.update_layout(showlegend=False)
        st.plotly_chart(fig_cp, use_container_width=True)

    # Margin & Quantity row
    col3, col4 = st.columns(2)
    with col3:
        fig_cm = px.bar(
            cat_perf.sort_values("Margin", ascending=True),
            x="Margin", y="Product_Category", orientation="h",
            title="Profit Margin % by Category", color="Product_Category",
            color_discrete_map=CATEGORY_COLORS,
            labels={"Margin": "Profit Margin (%)", "Product_Category": ""},
        )
        fig_cm = standard_layout(fig_cm)
        fig_cm.update_layout(showlegend=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col4:
        fig_cq = px.bar(
            cat_perf.sort_values("Quantity", ascending=True),
            x="Quantity", y="Product_Category", orientation="h",
            title="Quantity Sold by Category", color="Product_Category",
            color_discrete_map=CATEGORY_COLORS,
            labels={"Quantity": "Total Units Sold", "Product_Category": ""},
        )
        fig_cq = standard_layout(fig_cq)
        fig_cq.update_layout(showlegend=False)
        st.plotly_chart(fig_cq, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Top / Bottom Products ────────────────────────────────────────────
    st.markdown("## Product Performance")
    prod_perf = df.groupby("Product_Name").agg(
        Sales =("Total_Sales", "sum"),
        Profit=("Profit",      "sum"),
        Orders=("Order_ID",    "nunique"),
    ).reset_index()
    prod_perf["Margin"] = prod_perf["Profit"] / prod_perf["Sales"] * 100

    top10_sales  = prod_perf.nlargest(10, "Sales")
    top10_profit = prod_perf.nlargest(10, "Profit")
    bottom10     = prod_perf.nsmallest(10, "Profit")

    col5, col6 = st.columns(2)
    with col5:
        fig_ts = px.bar(
            top10_sales.sort_values("Sales"), x="Sales", y="Product_Name", orientation="h",
            title="Top 10 Products by Sales",
            color_discrete_sequence=[COLORS["secondary"]],
            labels={"Sales": "Total Sales ($)", "Product_Name": ""},
        )
        fig_ts = standard_layout(fig_ts, height=420)
        fig_ts.update_layout(showlegend=False)
        st.plotly_chart(fig_ts, use_container_width=True)

    with col6:
        fig_tp = px.bar(
            top10_profit.sort_values("Profit"), x="Profit", y="Product_Name", orientation="h",
            title="Top 10 Products by Profit",
            color_discrete_sequence=[COLORS["success"]],
            labels={"Profit": "Total Profit ($)", "Product_Name": ""},
        )
        fig_tp = standard_layout(fig_tp, height=420)
        fig_tp.update_layout(showlegend=False)
        st.plotly_chart(fig_tp, use_container_width=True)

    # Bottom performers
    st.markdown("### Bottom-Performing Products (by Profit)")
    fig_bot = px.bar(
        bottom10.sort_values("Profit"), x="Profit", y="Product_Name", orientation="h",
        title="10 Least Profitable Products",
        color_discrete_sequence=[COLORS["accent"]],
        labels={"Profit": "Total Profit ($)", "Product_Name": ""},
    )
    fig_bot = standard_layout(fig_bot, height=380)
    fig_bot.update_layout(showlegend=False)
    st.plotly_chart(fig_bot, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Monthly Sales Trend ──────────────────────────────────────────────
    st.markdown("## Monthly Sales Trend by Category")
    monthly_cat = (
        df.groupby(["YearMonth", "Product_Category"])["Total_Sales"]
        .sum().reset_index()
    )
    fig_mt = px.line(
        monthly_cat, x="YearMonth", y="Total_Sales",
        color="Product_Category", color_discrete_map=CATEGORY_COLORS,
        title="Monthly Sales by Product Category",
        labels={"Total_Sales": "Total Sales ($)", "YearMonth": "Month", "Product_Category": "Category"},
        markers=True,
    )
    fig_mt = standard_layout(fig_mt, height=380)
    fig_mt.update_xaxes(tickangle=45)
    st.plotly_chart(fig_mt, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Discount vs Profit ───────────────────────────────────────────────
    st.markdown("## Discount % vs Profit")
    col7, col8 = st.columns([3, 2])
    with col7:
        fig_dp = px.scatter(
            df, x="Discount_Percent", y="Profit",
            color="Product_Category", color_discrete_map=CATEGORY_COLORS,
            title="Discount % vs Profit per Order",
            labels={"Discount_Percent": "Discount (%)", "Profit": "Profit ($)"},
            opacity=0.55, size_max=8,
        )
        fig_dp.add_hline(y=0, line_dash="dash", line_color=COLORS["accent"], line_width=1)
        fig_dp = standard_layout(fig_dp, height=400)
        st.plotly_chart(fig_dp, use_container_width=True)

    with col8:
        # Bucket discounts into bands (use a local copy to avoid mutating the filtered df)
        bins   = [-1, 0, 10, 20, 30, 100]
        labels = ["0%", "1–10%", "11–20%", "21–30%", ">30%"]
        _df_band = df.copy()
        _df_band["Disc_Band"] = pd.cut(_df_band["Discount_Percent"], bins=bins, labels=labels)
        disc_band = _df_band.groupby("Disc_Band", observed=True).agg(
            Avg_Margin=("Profit_Margin", "mean"),
            Orders    =("Order_ID",      "count"),
        ).reset_index()
        fig_db = px.bar(
            disc_band, x="Disc_Band", y="Avg_Margin",
            title="Avg Profit Margin by Discount Band",
            color="Avg_Margin",
            color_continuous_scale=["#E84855", "#F4A261", "#3BB273"],
            labels={"Disc_Band": "Discount Band", "Avg_Margin": "Avg Margin (%)"},
        )
        fig_db = standard_layout(fig_db, height=400)
        fig_db.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_db, use_container_width=True)

    # Summary table
    st.markdown("### Category Summary Table")
    cat_display = cat_perf[["Product_Category", "Sales", "Profit", "Margin", "Orders", "Quantity"]].copy()
    cat_display["Sales"]  = cat_display["Sales"].apply(lambda x: f"${x:,.0f}")
    cat_display["Profit"] = cat_display["Profit"].apply(lambda x: f"${x:,.0f}")
    cat_display["Margin"] = cat_display["Margin"].apply(lambda x: f"{x:.1f}%")
    cat_display["Orders"] = cat_display["Orders"].apply(lambda x: f"{x:,}")
    cat_display["Quantity"] = cat_display["Quantity"].apply(lambda x: f"{x:,}")
    cat_display.columns = ["Category", "Total Sales", "Total Profit", "Margin %", "Orders", "Qty Sold"]
    st.dataframe(cat_display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CUSTOMER & REGIONAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_customer_regional(df: pd.DataFrame):
    st.markdown("# 👥 Customer & Regional Analysis")
    st.markdown("Segment behaviour, regional concentration, payment preferences, and customer value.")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Segment Performance ──────────────────────────────────────────────
    st.markdown("## Customer Segment Performance")
    seg_perf = df.groupby("Customer_Segment").agg(
        Sales   =("Total_Sales", "sum"),
        Profit  =("Profit",      "sum"),
        Orders  =("Order_ID",    "nunique"),
        Customers=("Customer_Name", "nunique"),
    ).reset_index()
    seg_perf["AOV"]    = seg_perf["Sales"] / seg_perf["Orders"]
    seg_perf["Margin"] = seg_perf["Profit"] / seg_perf["Sales"] * 100

    seg_colors = {"Consumer": "#2E86AB", "Corporate": "#3BB273", "Home Office": "#F4A261"}

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_ss = px.bar(
            seg_perf.sort_values("Sales"), x="Sales", y="Customer_Segment", orientation="h",
            title="Sales by Segment", color="Customer_Segment",
            color_discrete_map=seg_colors,
            labels={"Sales": "Total Sales ($)", "Customer_Segment": ""},
        )
        fig_ss = standard_layout(fig_ss)
        fig_ss.update_layout(showlegend=False)
        st.plotly_chart(fig_ss, use_container_width=True)

    with col2:
        fig_sp = px.bar(
            seg_perf.sort_values("Profit"), x="Profit", y="Customer_Segment", orientation="h",
            title="Profit by Segment", color="Customer_Segment",
            color_discrete_map=seg_colors,
            labels={"Profit": "Total Profit ($)", "Customer_Segment": ""},
        )
        fig_sp = standard_layout(fig_sp)
        fig_sp.update_layout(showlegend=False)
        st.plotly_chart(fig_sp, use_container_width=True)

    with col3:
        fig_aov = px.bar(
            seg_perf.sort_values("AOV"), x="AOV", y="Customer_Segment", orientation="h",
            title="Avg Order Value by Segment", color="Customer_Segment",
            color_discrete_map=seg_colors,
            labels={"AOV": "AOV ($)", "Customer_Segment": ""},
        )
        fig_aov = standard_layout(fig_aov)
        fig_aov.update_layout(showlegend=False)
        st.plotly_chart(fig_aov, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Segment × Category Heatmap ───────────────────────────────────────
    st.markdown("## Segment × Category Sales Heatmap")
    seg_cat = df.pivot_table(
        values="Total_Sales", index="Customer_Segment",
        columns="Product_Category", aggfunc="sum", fill_value=0,
    )
    fig_hm = px.imshow(
        seg_cat, text_auto=".0f",
        color_continuous_scale="Blues",
        title="Total Sales ($) — Segment × Category",
        labels=dict(x="Product Category", y="Customer Segment", color="Sales ($)"),
        aspect="auto",
    )
    fig_hm.update_layout(height=280, margin=dict(l=30, r=30, t=40, b=30),
                         paper_bgcolor="white", font=dict(family="Segoe UI, Arial", size=12))
    st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Regional Analysis ────────────────────────────────────────────────
    st.markdown("## Regional Performance")
    reg_perf = df.groupby("Region").agg(
        Sales   =("Total_Sales", "sum"),
        Profit  =("Profit",      "sum"),
        Orders  =("Order_ID",    "nunique"),
    ).reset_index()
    reg_perf["Margin"] = reg_perf["Profit"] / reg_perf["Sales"] * 100

    col4, col5 = st.columns(2)
    with col4:
        fig_rs = px.bar(
            reg_perf.sort_values("Sales"), x="Sales", y="Region", orientation="h",
            title="Sales by Region", color="Region",
            color_discrete_sequence=REGION_COLORS,
            labels={"Sales": "Total Sales ($)", "Region": ""},
        )
        fig_rs = standard_layout(fig_rs)
        fig_rs.update_layout(showlegend=False)
        st.plotly_chart(fig_rs, use_container_width=True)

    with col5:
        fig_rp = px.bar(
            reg_perf.sort_values("Profit"), x="Profit", y="Region", orientation="h",
            title="Profit by Region", color="Region",
            color_discrete_sequence=REGION_COLORS,
            labels={"Profit": "Total Profit ($)", "Region": ""},
        )
        fig_rp = standard_layout(fig_rp)
        fig_rp.update_layout(showlegend=False)
        st.plotly_chart(fig_rp, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Country Performance ──────────────────────────────────────────────
    st.markdown("## Country Performance")
    country_perf = df.groupby("Country").agg(
        Sales =("Total_Sales", "sum"),
        Profit=("Profit",      "sum"),
        Orders=("Order_ID",    "nunique"),
    ).reset_index()
    country_perf["Margin"] = country_perf["Profit"] / country_perf["Sales"] * 100
    country_perf = country_perf.sort_values("Sales", ascending=False)

    fig_cc = px.bar(
        country_perf, x="Country", y="Sales",
        title="Sales by Country", color="Sales",
        color_continuous_scale="Blues",
        labels={"Sales": "Total Sales ($)", "Country": "Country"},
    )
    fig_cc = standard_layout(fig_cc, height=380)
    fig_cc.update_xaxes(tickangle=45)
    fig_cc.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_cc, use_container_width=True)

    # Country table
    country_disp = country_perf[["Country", "Sales", "Profit", "Margin", "Orders"]].copy()
    country_disp["Sales"]  = country_disp["Sales"].apply(lambda x: f"${x:,.0f}")
    country_disp["Profit"] = country_disp["Profit"].apply(lambda x: f"${x:,.0f}")
    country_disp["Margin"] = country_disp["Margin"].apply(lambda x: f"{x:.1f}%")
    country_disp.columns   = ["Country", "Total Sales", "Total Profit", "Margin %", "Orders"]
    st.dataframe(country_disp, use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Payment Method ───────────────────────────────────────────────────
    st.markdown("## Payment Method Distribution")
    pay_perf = df.groupby("Payment_Method").agg(
        Sales  =("Total_Sales", "sum"),
        Orders =("Order_ID",    "nunique"),
        Profit =("Profit",      "sum"),
    ).reset_index()

    col6, col7 = st.columns(2)
    with col6:
        fig_pay = px.pie(
            pay_perf, values="Orders", names="Payment_Method",
            title="Orders by Payment Method",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4,
        )
        fig_pay = standard_layout(fig_pay)
        fig_pay.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pay, use_container_width=True)

    with col7:
        fig_pays = px.bar(
            pay_perf.sort_values("Sales"), x="Sales", y="Payment_Method", orientation="h",
            title="Sales by Payment Method", color="Payment_Method",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={"Sales": "Total Sales ($)", "Payment_Method": ""},
        )
        fig_pays = standard_layout(fig_pays)
        fig_pays.update_layout(showlegend=False)
        st.plotly_chart(fig_pays, use_container_width=True)

    # ── Customer Insights ────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 🤖 Customer & Regional Insights")
    st.caption("Rule-based insights computed from the filtered dataset.")

    best_seg   = seg_perf.loc[seg_perf["Profit"].idxmax(),  "Customer_Segment"]
    best_seg_m = seg_perf.loc[seg_perf["Customer_Segment"] == best_seg, "Margin"].values[0]
    best_reg   = reg_perf.loc[reg_perf["Sales"].idxmax(), "Region"]
    low_reg    = reg_perf.loc[reg_perf["Sales"].idxmin(), "Region"]
    low_reg_s  = reg_perf.loc[reg_perf["Region"] == low_reg, "Sales"].values[0]
    best_country = country_perf.iloc[0]["Country"]
    top_payment  = pay_perf.loc[pay_perf["Orders"].idxmax(), "Payment_Method"]

    total_sales = df["Total_Sales"].sum()

    insights_html = ""
    insights_html += insight_box(
        "Top Customer Segment",
        f"<b>{best_seg}</b> is the most profitable customer segment in the filtered dataset, with a profit margin of "
        f"<b>{fmt_pct(best_seg_m)}</b>. Targeted offers and upsell strategies may benefit from prioritising this group."
    )
    insights_html += insight_box(
        "Regional Revenue Leader",
        f"<b>{best_reg}</b> leads all regions in total sales. "
        f"<b>{low_reg}</b> is the lowest-revenue region in the current dataset at <b>{fmt_currency(low_reg_s)}</b> "
        f"({fmt_pct(low_reg_s / total_sales * 100 if total_sales else 0)} of total)."
    )
    insights_html += insight_box(
        "Top Country",
        f"<b>{best_country}</b> is the single largest country by sales within the filtered data."
    )
    insights_html += insight_box(
        "Preferred Payment Method",
        f"<b>{top_payment}</b> is the most frequently used payment method by order count, "
        "which may influence checkout optimisation priorities."
    )
    st.markdown(insights_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — RISK, OPPORTUNITY & ACTION
# ─────────────────────────────────────────────────────────────────────────────
def page_risk_opportunity(df: pd.DataFrame):
    st.markdown("# ⚡ Risk, Opportunity & Recommended Action")
    st.markdown(
        "Decision-support analysis derived entirely from calculated values in the filtered dataset. "
        "No external assumptions are made."
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    total_sales  = df["Total_Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = len(df)

    # ── Pre-compute all metrics ───────────────────────────────────────────

    # Loss-making orders
    loss_df = df[df["Is_Loss"]].copy()
    loss_count  = len(loss_df)
    loss_pct    = loss_count / total_orders * 100 if total_orders else 0
    loss_value  = loss_df["Profit"].sum()

    # Category profitability
    cat_perf = df.groupby("Product_Category").agg(
        Sales =("Total_Sales", "sum"),
        Profit=("Profit",      "sum"),
    ).reset_index()
    cat_perf["Margin"] = cat_perf["Profit"] / cat_perf["Sales"] * 100
    low_margin_cat = cat_perf.nsmallest(1, "Margin").iloc[0]
    best_margin_cat = cat_perf.nlargest(1, "Margin").iloc[0]
    best_profit_cat = cat_perf.nlargest(1, "Profit").iloc[0]

    # Discount impact
    high_disc_df = df[df["Discount_Percent"] >= 20]
    low_disc_df  = df[df["Discount_Percent"] <  20]
    high_disc_pct   = len(high_disc_df) / total_orders * 100 if total_orders else 0
    avg_m_high_disc = high_disc_df["Profit_Margin"].mean() if not high_disc_df.empty else 0
    avg_m_low_disc  = low_disc_df["Profit_Margin"].mean()  if not low_disc_df.empty  else 0
    disc_margin_gap = avg_m_low_disc - avg_m_high_disc

    # Revenue concentration
    reg_sales   = df.groupby("Region")["Total_Sales"].sum().sort_values(ascending=False)
    top_reg     = reg_sales.index[0]
    top_reg_pct = reg_sales.iloc[0] / total_sales * 100 if total_sales else 0
    low_reg     = reg_sales.index[-1]
    low_reg_pct = reg_sales.iloc[-1] / total_sales * 100 if total_sales else 0
    low_reg_s   = reg_sales.iloc[-1]

    # Shipping cost vs profit
    shipping_loss_df = df[df["Shipping_Cost"] > df["Profit"]]
    ship_pct = len(shipping_loss_df) / total_orders * 100 if total_orders else 0

    # High-profit products
    prod_profit = df.groupby("Product_Name")["Profit"].sum().sort_values(ascending=False)
    top3_prods  = prod_profit.head(3)

    # Segment opportunity
    seg_perf = df.groupby("Customer_Segment").agg(
        Sales =("Total_Sales", "sum"),
        Profit=("Profit",      "sum"),
    ).reset_index()
    seg_perf["Margin"] = seg_perf["Profit"] / seg_perf["Sales"] * 100
    best_seg = seg_perf.nlargest(1, "Profit").iloc[0]

    # ─────────────────────────────────────────────────────────────────────
    # RISKS
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("## 🔴 Identified Risks")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown(risk_card(
            "Loss-Making Orders",
            f"{loss_count:,} orders ({fmt_pct(loss_pct)} of all orders in the filtered data) "
            f"have negative profit, representing a total loss of {fmt_currency(abs(loss_value))}.",
            "A meaningful share of transactions are destroying value. These may result from "
            "excessive discounting, high shipping costs, or low-margin product combinations.",
            "Continued loss-making orders erode overall profitability and can mask the true "
            "performance of profitable segments if not tracked separately.",
            "Identify the common product categories, discount levels, and regions associated "
            "with loss orders. Introduce order-level margin thresholds as part of pricing approvals.",
        ), unsafe_allow_html=True)

        st.markdown(risk_card(
            "Low-Margin Category",
            f"{low_margin_cat['Product_Category']} has the lowest profit margin at "
            f"{fmt_pct(low_margin_cat['Margin'])} on total sales of {fmt_currency(low_margin_cat['Sales'])}.",
            "This category is generating revenue but relatively little profit. "
            "Pricing levels or discount rates within this category may be contributing to the lower margin.",
            "If this category represents significant revenue, poor margins translate into a "
            "disproportionately large drag on total profitability.",
            "Review unit pricing and discount policies within this category to assess "
            "whether margin improvement is achievable through price or discount adjustments.",
        ), unsafe_allow_html=True)

    with col_r2:
        if high_disc_pct > 0:
            st.markdown(risk_card(
                "High Discount Impact on Margin",
                f"{fmt_pct(high_disc_pct)} of orders carry a discount of 20 % or more. "
                f"These orders have an average margin of {fmt_pct(avg_m_high_disc)}, compared to "
                f"{fmt_pct(avg_m_low_disc)} for orders below 20 % discount — "
                f"a gap of {fmt_pct(disc_margin_gap)} in margin.",
                "Heavy discounting is associated with materially lower profit margins. "
                "Discounts at this level may not be offset by sufficiently higher order volumes.",
                "Uncapped or poorly targeted discounting can erode profitability across the "
                "entire product portfolio, particularly in already low-margin categories.",
                "Establish maximum discount thresholds by product category. Require management "
                "approval for discounts above 20 %. Evaluate whether volume uplift justifies margin give-up.",
            ), unsafe_allow_html=True)

        if top_reg_pct > 40:
            st.markdown(risk_card(
                "Revenue Concentration Risk",
                f"{top_reg} contributes {fmt_pct(top_reg_pct)} of total revenue in the "
                f"filtered dataset. The lowest region, {low_reg}, contributes only {fmt_pct(low_reg_pct)}.",
                "A large share of revenue is concentrated in a single region. "
                "This means that any decline in that region's orders would have an outsized effect on total revenue.",
                "High regional concentration reduces the portfolio's revenue diversity "
                "and increases exposure to single-region order volume fluctuations.",
                "Develop targeted growth plans for lower-revenue regions. Set regional "
                "revenue diversification targets and track them quarterly.",
            ), unsafe_allow_html=True)

        if ship_pct > 5:
            st.markdown(risk_card(
                "Shipping Cost Exceeding Profit",
                f"{fmt_pct(ship_pct)} of orders have a shipping cost that exceeds the reported "
                "profit on that order.",
                "For these orders, shipping cost exceeds the reported profit, indicating that "
                "logistics costs are consuming a substantial portion of order-level value.",
                "A high proportion of orders where shipping cost exceeds profit may reduce "
                "net order-level returns, particularly for lower-value orders.",
                "Introduce a minimum order value threshold for free or discounted shipping. "
                "Review fulfilment strategies for small orders — consider consolidation or "
                "local pick-up options.",
            ), unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # OPPORTUNITIES
    # ─────────────────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 🟢 Identified Opportunities")

    col_o1, col_o2 = st.columns(2)

    with col_o1:
        st.markdown(opp_card(
            "High-Profit Category Leadership",
            f"{best_profit_cat['Product_Category']} generates the highest absolute profit "
            f"({fmt_currency(best_profit_cat['Profit'])}) with a margin of "
            f"{fmt_pct(best_profit_cat['Margin'])}.",
            f"{best_profit_cat['Product_Category']} is the largest absolute profit contributor "
            "in the current portfolio.",
            "Increasing the revenue share of this category — through promotion or product range "
            "expansion — could improve overall portfolio profit.",
            "Allocate additional marketing spend to this category. "
            "Review the product mix within it to prioritise the highest-margin SKUs.",
        ), unsafe_allow_html=True)

        st.markdown(opp_card(
            "Top Customer Segment Expansion",
            f"{best_seg['Customer_Segment']} is the most profitable customer segment, "
            f"with total profit of {fmt_currency(best_seg['Profit'])} "
            f"and a profit margin of {fmt_pct(best_seg['Margin'])}.",
            "This segment generates the highest total profit and profit margin among the "
            "customer segments in the filtered dataset.",
            "Deeper penetration of this segment — through targeted offers and account "
            "management — could yield disproportionate profit growth.",
            "Develop a segment-specific retention programme. Analyse the product mix "
            "purchased by this segment and identify cross-sell opportunities.",
        ), unsafe_allow_html=True)

    with col_o2:
        st.markdown(opp_card(
            "Regional Growth Opportunity",
            f"{low_reg} is the lowest-revenue region with {fmt_currency(low_reg_s)} "
            f"in sales ({fmt_pct(low_reg_pct)} of total).",
            "This region currently contributes the smallest share of revenue in the dataset. "
            "The gap between it and the highest-revenue region represents a potential area "
            "for increased focus.",
            "A lower-revenue region may present room for revenue growth if additional "
            "commercial focus is applied, though the dataset does not establish the cause "
            "of the current revenue level.",
            "Investigate the product mix and order patterns in this region using the available data. "
            "Consider setting measurable revenue targets for this region and tracking progress.",
        ), unsafe_allow_html=True)

        st.markdown(opp_card(
            "Discount Optimisation",
            f"Orders with discount below 20 % show an average profit margin of "
            f"{fmt_pct(avg_m_low_disc)}, compared to {fmt_pct(avg_m_high_disc)} "
            f"for orders at 20 % or above.",
            "Higher-discount orders show lower average profit margins in this dataset. "
            "Reducing the proportion of high-discount orders could improve overall margin, "
            "though the effect on order volumes cannot be determined from this dataset alone.",
            "Reducing average discount levels could improve overall portfolio profit margin, "
            "based on the margin differential observed in the current data.",
            "Review the discount approval process and consider introducing tiered discount caps "
            "by segment and category. Analyse which order types or segments drive the highest "
            "discount usage.",
        ), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Loss-Making Orders Detail ─────────────────────────────────────────
    st.markdown("## Loss-Making Orders — Detail")
    st.caption(f"{loss_count:,} orders with negative profit. Showing up to 200 rows.")
    if not loss_df.empty:
        display_cols = ["Order_ID", "Order_Date", "Customer_Segment", "Country", "Region",
                        "Product_Category", "Product_Name", "Discount_Percent",
                        "Total_Sales", "Shipping_Cost", "Profit"]
        loss_show = loss_df[display_cols].copy()
        loss_show["Order_Date"]      = loss_show["Order_Date"].dt.strftime("%Y-%m-%d")
        loss_show["Total_Sales"]     = loss_show["Total_Sales"].apply(lambda x: f"${x:,.2f}")
        loss_show["Shipping_Cost"]   = loss_show["Shipping_Cost"].apply(lambda x: f"${x:,.2f}")
        loss_show["Profit"]          = loss_show["Profit"].apply(lambda x: f"${x:,.2f}")
        loss_show["Discount_Percent"] = loss_show["Discount_Percent"].apply(lambda x: f"{x:.0f}%")
        st.dataframe(loss_show.head(200), use_container_width=True, hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Summary Chart: Profit Margin by Category × Segment ───────────────
    st.markdown("## Profit Margin by Category & Segment")
    cs_margin = df.groupby(["Product_Category", "Customer_Segment"]).apply(
        lambda g: pd.Series({
            "Margin": (g["Profit"].sum() / g["Total_Sales"].sum() * 100) if g["Total_Sales"].sum() != 0 else 0
        })
    ).reset_index()
    fig_csm = px.bar(
        cs_margin, x="Product_Category", y="Margin", color="Customer_Segment",
        barmode="group", title="Profit Margin % by Category and Customer Segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"Margin": "Profit Margin (%)", "Product_Category": "Category", "Customer_Segment": "Segment"},
    )
    fig_csm = standard_layout(fig_csm, height=380)
    st.plotly_chart(fig_csm, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION & MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Sidebar filters — applied globally
    filtered_df = build_sidebar(RAW_DF)

    # Page navigation via selectbox in main area header
    st.markdown(
        "<div style='background:#1F3A5F;padding:0.7rem 1.2rem;border-radius:8px;margin-bottom:1rem;'>"
        "<span style='color:white;font-size:1.25rem;font-weight:700;'>📊 E-Commerce BI Dashboard</span>"
        "<span style='color:#93C5FD;font-size:0.85rem;margin-left:1rem;'>"
        "AI-Powered Business Intelligence &amp; Decision Support</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    pages = [
        "📊 Executive Overview",
        "📦 Sales & Product Analysis",
        "👥 Customer & Regional Analysis",
        "⚡ Risk, Opportunity & Action",
    ]
    selected_page = st.selectbox("Navigate to page:", pages, label_visibility="collapsed")

    st.markdown("---")

    if selected_page == pages[0]:
        page_executive_overview(filtered_df)
    elif selected_page == pages[1]:
        page_sales_product(filtered_df)
    elif selected_page == pages[2]:
        page_customer_regional(filtered_df)
    elif selected_page == pages[3]:
        page_risk_opportunity(filtered_df)

    # Footer
    st.markdown(
        "<br><div style='text-align:center;color:#9CA3AF;font-size:0.78rem;"
        "border-top:1px solid #E5E7EB;padding-top:0.8rem;margin-top:1.5rem;'>"
        "AI-Powered E-Commerce BI Dashboard &nbsp;|&nbsp; Data: global_ecommerce_sales.csv &nbsp;|&nbsp; "
        "Built with Streamlit &amp; Plotly"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
