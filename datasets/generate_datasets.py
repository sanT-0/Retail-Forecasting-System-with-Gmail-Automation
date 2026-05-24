"""
     Future Retail Sales Forecasting  Dataset Generator
     Created by: sanT
     Project: Future Retail Sales Forecasting Using Machine Learning

Generates realistic Indian retail sales datasets with clear trends,
seasonality, and festival effects — essential for high R² scores.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── Seed for reproducibility ──────────────────────────────────────────────────
np.random.seed(42)

# ── Indian Retail Context ─────────────────────────────────────────────────────
INDIAN_CITIES      = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
                       "Coimbatore", "Salem", "Cuddalore", "Chidambaram", "Pondicherry"]
STORE_TYPES        = ["Supermarket", "Hypermarket", "Convenience Store", "Department Store"]
PAYMENT_METHODS    = ["Cash", "UPI", "Credit Card", "Debit Card", "Net Banking"]

# ── Shared date range ─────────────────────────────────────────────────────────
START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2026, 4, 15)
DATE_RANGE = pd.date_range(START_DATE, END_DATE, freq="D")

# ── Festival calendar (month, day) ────────────────────────────────────────────
FESTIVALS = {
    "Diwali":        (10, 24),
    "Holi":          (3,  25),
    "Eid":           (4,  10),
    "Dussehra":      (10, 12),
    "Navratri":      (10,  3),
    "Christmas":     (12, 25),
    "New Year":      (1,   1),
    "Independence":  (8,  15),
    "Republic Day":  (1,  26),
    "Raksha Bandhan":(8,  19),
}


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL BUILDERS  (shared by all generators)
# ─────────────────────────────────────────────────────────────────────────────

def _growth_trend(dates: pd.DatetimeIndex, annual_rate: float = 0.15) -> np.ndarray:
    """Smooth compounding year-over-year growth."""
    days_elapsed = (dates - dates[0]).days
    return (1 + annual_rate) ** (days_elapsed / 365.25)


def _annual_seasonality(dates: pd.DatetimeIndex,
                         peak_month: int = 10,
                         amplitude: float = 0.30) -> np.ndarray:
    """
    Smooth sinusoidal annual seasonality.
    Peak month = month with highest sales (default Oct, festive season).
    """
    day_of_year = dates.dayofyear
    peak_day    = (peak_month - 1) * 30 + 15          # approx peak day-of-year
    return 1 + amplitude * np.sin(2 * np.pi * (day_of_year - peak_day) / 365.25)


def _weekly_seasonality(dates: pd.DatetimeIndex,
                         weekend_boost: float = 0.18) -> np.ndarray:
    """Saturday/Sunday uplift."""
    is_weekend = (dates.dayofweek >= 5).astype(float)
    return 1 + is_weekend * weekend_boost


def _monthly_payday(dates: pd.DatetimeIndex, boost: float = 0.12) -> np.ndarray:
    """Salary-day spending spike: 1st and last 2 days of month."""
    is_payday = ((dates.day <= 2) | (dates.day >= 28)).astype(float)
    return 1 + is_payday * boost


def _festival_multiplier(dates: pd.DatetimeIndex,
                          peak_boost: float = 0.60,
                          category_multiplier: float = 1.0) -> np.ndarray:
    """
    Vectorised festival proximity multiplier.
    Within 7 days → full boost; within 14 days → half boost.
    """
    boost = np.ones(len(dates))
    for (month, day) in FESTIVALS.values():
        for year in range(START_DATE.year - 1, END_DATE.year + 2):
            try:
                fest = pd.Timestamp(year, month, day)
            except ValueError:
                continue
            diff = np.abs((dates - fest).days)
            close  = diff <= 7
            medium = (diff > 7) & (diff <= 14)
            boost[close]  += category_multiplier * peak_boost * (1 - diff[close]  / 14)
            boost[medium] += category_multiplier * peak_boost * 0.25 * (1 - (diff[medium] - 7) / 14)
    return boost


def _gaussian_noise(n: int, sigma: float = 0.08) -> np.ndarray:
    """Multiplicative Gaussian noise centered at 1.0."""
    return np.random.normal(1.0, sigma, n).clip(0.5, 2.5)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: Pharmacy Daily Sales (Healthcare)
# ─────────────────────────────────────────────────────────────────────────────
def generate_pharmacy_dataset() -> pd.DataFrame:
    """
    Daily pharmacy chain revenue.
    Clear trend + monsoon/winter seasonal peaks + modest festival boost.
    """
    dates = DATE_RANGE
    n     = len(dates)

    base_rev = 180_000               # daily chain-level base (₹ 1.8 L/day)

    trend    = _growth_trend(dates, annual_rate=0.12)
    seasonal = (
        1 + 0.25 * np.sin(2 * np.pi * (dates.dayofyear - 200) / 365.25)   # Monsoon peak (~Jul)
        + 0.12 * np.sin(2 * np.pi * (dates.dayofyear - 355) / 365.25)     # Winter peak (~Dec)
    )
    weekend  = _weekly_seasonality(dates, weekend_boost=0.08)
    festival = _festival_multiplier(dates, peak_boost=0.20, category_multiplier=0.8)
    noise    = _gaussian_noise(n, sigma=0.07)

    revenue  = base_rev * trend * seasonal * weekend * festival * noise

    med_types = ["Prescription", "OTC", "Wellness", "Personal Care", "Medical Equipment"]
    cities    = np.random.choice(INDIAN_CITIES, n)

    df = pd.DataFrame({
        "Date":         dates.strftime("%Y-%m-%d"),
        "City":         cities,
        "Type":         np.random.choice(med_types, n),
        "Revenue_INR":  np.round(revenue, 2),
        "Orders":       (revenue / np.random.uniform(200, 800, n)).astype(int),
        "Stock_Out_Flag": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: Electronics Gadgets (High-Value)
# ─────────────────────────────────────────────────────────────────────────────
def generate_electronics_dataset() -> pd.DataFrame:
    """
    Daily electronics revenue — strong Diwali/New Year spikes + growth trend.
    """
    dates = DATE_RANGE
    n     = len(dates)

    base_rev = 350_000              # ₹ 3.5 L / day

    trend    = _growth_trend(dates, annual_rate=0.18)   # fast-growing category
    seasonal = _annual_seasonality(dates, peak_month=10, amplitude=0.40)
    weekend  = _weekly_seasonality(dates, weekend_boost=0.22)
    payday   = _monthly_payday(dates, boost=0.15)
    festival = _festival_multiplier(dates, peak_boost=0.80, category_multiplier=1.5)
    noise    = _gaussian_noise(n, sigma=0.09)

    revenue  = base_rev * trend * seasonal * weekend * payday * festival * noise

    gadgets  = ["Smartphones", "Laptops", "Tablets", "Smartwatches", "Headphones"]
    cities   = np.random.choice(INDIAN_CITIES, n)

    df = pd.DataFrame({
        "Order_Date":       dates.strftime("%Y-%m-%d"),
        "Product_Category": np.random.choice(gadgets, n),
        "City":             cities,
        "Revenue_INR":      np.round(revenue, 2),
        "EMI_Option":       np.random.choice([0, 1], n, p=[0.4, 0.6]),
        "Warranty_Ext":     np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 3: Grocery App Delivery (Hyper-local)
# ─────────────────────────────────────────────────────────────────────────────
def generate_delivery_dataset() -> pd.DataFrame:
    """
    Daily grocery delivery revenue — strong weekend + monthly payday peaks.
    """
    dates = DATE_RANGE
    n     = len(dates)

    base_rev = 120_000             # ₹ 1.2 L / day

    trend    = _growth_trend(dates, annual_rate=0.25)   # hyper-growth startup
    seasonal = _annual_seasonality(dates, peak_month=12, amplitude=0.20)
    weekend  = _weekly_seasonality(dates, weekend_boost=0.30)
    payday   = _monthly_payday(dates, boost=0.20)
    festival = _festival_multiplier(dates, peak_boost=0.50, category_multiplier=1.2)
    noise    = _gaussian_noise(n, sigma=0.10)

    revenue  = base_rev * trend * seasonal * weekend * payday * festival * noise

    cities   = np.random.choice(INDIAN_CITIES, n)

    # Delivery fee and tip are small compared to order total
    del_fees = np.random.uniform(10, 60, n)
    tips     = np.random.choice([0, 10, 20, 50], n, p=[0.5, 0.3, 0.15, 0.05])

    df = pd.DataFrame({
        "Delivery_Date":    dates.strftime("%Y-%m-%d"),
        "City":             cities,
        "Revenue_INR":      np.round(revenue + del_fees + tips, 2),
        "Delivery_Fee":     np.round(del_fees, 2),
        "Tip_Amount":       tips.astype(float),
        "Delivery_Time_Mins": np.random.randint(10, 60, n),
        "Payment_App":      np.random.choice(["Paytm", "PhonePe", "GPay", "AmazonPay"], n),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 4: Furniture & Decor (Seasonal — Weekly)
# ─────────────────────────────────────────────────────────────────────────────
def generate_furniture_dataset() -> pd.DataFrame:
    """
    Weekly furniture & home-decor revenue.
    Very strong Diwali double-boost + steady growth.
    """
    dates = pd.date_range(START_DATE, END_DATE, freq="W")
    n     = len(dates)

    base_rev = 900_000              # ₹ 9 L / week

    trend    = _growth_trend(dates, annual_rate=0.14)
    seasonal = _annual_seasonality(dates, peak_month=10, amplitude=0.50)
    festival = _festival_multiplier(dates, peak_boost=1.20, category_multiplier=2.0)
    noise    = _gaussian_noise(n, sigma=0.10)

    revenue  = base_rev * trend * seasonal * festival * noise

    items    = ["Sofas", "Dining Tables", "Beds", "Wall Decor", "Lamps"]
    cities   = np.random.choice(INDIAN_CITIES, n)

    df = pd.DataFrame({
        "Week_End":       dates.strftime("%Y-%m-%d"),
        "Category":       np.random.choice(items, n),
        "City":           cities,
        "Revenue_INR":    np.round(revenue, 2),
        "Lead_Time_Days": np.random.randint(7, 45, n),
        "Materials":      np.random.choice(["Wood", "Metal", "Fabric", "Marble"], n),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 5: Luxury Brand (Elite High-Margin)
# ─────────────────────────────────────────────────────────────────────────────
def generate_luxury_dataset() -> pd.DataFrame:
    """
    Daily luxury brand revenue — slower growth, clean trend, low noise.
    High-ticket items → strong festival sensitivity.
    """
    dates = DATE_RANGE
    n     = len(dates)

    base_rev = 95_000              # ₹ 95K / day (premium, concentrated)

    trend    = _growth_trend(dates, annual_rate=0.10)
    seasonal = _annual_seasonality(dates, peak_month=11, amplitude=0.35)
    payday   = _monthly_payday(dates, boost=0.18)
    festival = _festival_multiplier(dates, peak_boost=0.90, category_multiplier=1.8)
    noise    = _gaussian_noise(n, sigma=0.06)   # lower noise → cleaner signal

    revenue  = base_rev * trend * seasonal * payday * festival * noise

    brands   = ["Luxury Watches", "Premium Handbags", "Designer Wear", "Rare Scents"]
    cities   = np.random.choice(["Mumbai", "Delhi", "Bengaluru"], n)

    df = pd.DataFrame({
        "Sale_Date":      dates.strftime("%Y-%m-%d"),
        "Luxury_Item":    np.random.choice(brands, n),
        "City":           cities,
        "Revenue_INR":    np.round(revenue, 2),
        "Customer_Tier":  np.random.choice(["Gold", "Platinum", "VVIP"], n, p=[0.7, 0.2, 0.1]),
        "Margin_Pct":     np.round(np.random.uniform(40, 75, n), 2),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 6: Automotive Spare Parts (Steady Growth)
# ─────────────────────────────────────────────────────────────────────────────
def generate_automotive_dataset() -> pd.DataFrame:
    """Daily auto parts revenue — low noise, steady trend, high predictability."""
    dates = DATE_RANGE
    n = len(dates)
    base_rev = 240_000
    trend = _growth_trend(dates, annual_rate=0.08)
    seasonal = _annual_seasonality(dates, peak_month=5, amplitude=0.15)
    weekend = _weekly_seasonality(dates, weekend_boost=0.05)
    noise = _gaussian_noise(n, sigma=0.04)
    revenue = base_rev * trend * seasonal * weekend * noise
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Part_Category": np.random.choice(["Engine", "Brakes", "Suspension", "Electrical", "Body"], n),
        "City": np.random.choice(INDIAN_CITIES, n),
        "Revenue_INR": np.round(revenue, 2),
        "B2B_Client": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 7: Fashion & Apparel (High Seasonality)
# ─────────────────────────────────────────────────────────────────────────────
def generate_fashion_dataset() -> pd.DataFrame:
    """Daily fashion revenue — strong Summer/Winter cycles."""
    dates = DATE_RANGE
    n = len(dates)
    base_rev = 150_000
    trend = _growth_trend(dates, annual_rate=0.20)
    # Peak in May (Summer) and Nov (Winter)
    seasonal = (1 + 0.3 * np.sin(2 * np.pi * (dates.dayofyear - 135)/365.25) + 
                0.25 * np.sin(2 * np.pi * (dates.dayofyear - 320)/365.25))
    festival = _festival_multiplier(dates, peak_boost=0.70)
    noise = _gaussian_noise(n, sigma=0.06)
    revenue = base_rev * trend * seasonal * festival * noise
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Apparel_Type": np.random.choice(["Menswear", "Womenswear", "Kids", "Ethnic", "Athleisure"], n),
        "Revenue_INR": np.round(revenue, 2),
        "Online_Sale": np.random.choice([0, 1], n),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 8: Beauty & Cosmetics (Payday Loyalty)
# ─────────────────────────────────────────────────────────────────────────────
def generate_beauty_dataset() -> pd.DataFrame:
    """Daily beauty sales — strong payday and weekend correlations."""
    dates = DATE_RANGE
    n = len(dates)
    base_rev = 80_000
    trend = _growth_trend(dates, annual_rate=0.15)
    payday = _monthly_payday(dates, boost=0.25)
    weekend = _weekly_seasonality(dates, weekend_boost=0.15)
    noise = _gaussian_noise(n, sigma=0.05)
    revenue = base_rev * trend * payday * weekend * noise
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Category": np.random.choice(["Skincare", "Makeup", "Fragrance", "Haircare", "Bath"], n),
        "Revenue_INR": np.round(revenue, 2),
        "Repeat_Customer": np.random.choice([0, 1], n, p=[0.4, 0.6]),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 9: Home Appliances (Summer Peak)
# ─────────────────────────────────────────────────────────────────────────────
def generate_appliances_dataset() -> pd.DataFrame:
    """Daily appliances revenue — extreme climate-driven seasonality (Summer)."""
    dates = DATE_RANGE
    n = len(dates)
    base_rev = 420_000
    trend = _growth_trend(dates, annual_rate=0.12)
    # Strong Summer peak in April/May
    seasonal = _annual_seasonality(dates, peak_month=5, amplitude=0.60)
    festival = _festival_multiplier(dates, peak_boost=0.50)
    noise = _gaussian_noise(n, sigma=0.06)
    revenue = base_rev * trend * seasonal * festival * noise
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Item": np.random.choice(["Air Conditioner", "Refrigerator", "Washing Machine", "Oven", "Vacuums"], n),
        "Revenue_INR": np.round(revenue, 2),
        "Installation_Req": np.random.choice([0, 1], n, p=[0.3, 0.7]),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 10: Toys & Games (Festive Weekly)
# ─────────────────────────────────────────────────────────────────────────────
def generate_toys_dataset() -> pd.DataFrame:
    """Weekly toys revenue — massive Q4 and holiday spikes."""
    dates = pd.date_range(START_DATE, END_DATE, freq="W")
    n = len(dates)
    base_rev = 120_000
    trend = _growth_trend(dates, annual_rate=0.10)
    seasonal = _annual_seasonality(dates, peak_month=12, amplitude=0.40)
    festival = _festival_multiplier(dates, peak_boost=1.50, category_multiplier=2.5)
    noise = _gaussian_noise(n, sigma=0.08)
    revenue = base_rev * trend * seasonal * festival * noise
    df = pd.DataFrame({
        "Week_End": dates.strftime("%Y-%m-%d"),
        "Toy_Type": np.random.choice(["Action Figures", "Puzzles", "Electronic", "Outdoor", "Soft Toys"], n),
        "Revenue_INR": np.round(revenue, 2),
        "Age_Group": np.random.choice(["0-3", "4-7", "8-12", "13+"], n),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_datasets(out_dir: str = ".") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    print("\n Generating 10 Indian Retail Datasets (structured time series)...\n")
    datasets = {
        "pharmacy_daily":      generate_pharmacy_dataset(),
        "electronics_gadgets": generate_electronics_dataset(),
        "grocery_delivery":    generate_delivery_dataset(),
        "furniture_decor":     generate_furniture_dataset(),
        "luxury_brand":        generate_luxury_dataset(),
        "automotive_parts":    generate_automotive_dataset(),
        "fashion_apparel":     generate_fashion_dataset(),
        "beauty_cosmetics":    generate_beauty_dataset(),
        "home_appliances":     generate_appliances_dataset(),
        "toys_games":          generate_toys_dataset(),
    }
    for name, df in datasets.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"     Saved  {path}  ({len(df):,} rows)")
    print("\n All 10 datasets generated successfully!\n")
    return datasets


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generate_all_datasets(out_dir=base_dir)
