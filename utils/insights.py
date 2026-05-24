"""

  Future Retail Sales Forecasting  Strategic Insights Generator   
  Created by: sanT                                                

"""

import numpy as np
import pandas as pd

def get_strategic_recommendations(results: dict) -> list:
    """
    Analyze forecast results and return a list of actionable insights.
    
    Args:
        results (dict): Dictionary containing series, future_preds, future_dates, etc.
        
    Returns:
        list: List of recommendation dictionaries with 'category', 'icon', 'title', and 'description'.
    """
    series = results.get("series")
    future_preds = results.get("future_preds")
    future_dates = results.get("future_dates")
    
    if series is None or future_preds is None or len(future_preds) == 0:
        return []

    recommendations = []
    
    # 1. Analyze Trend
    avg_hist = np.mean(series)
    avg_fut = np.mean(future_preds)
    pct_change = ((avg_fut - avg_hist) / avg_hist) * 100 if avg_hist > 0 else 0
    
    # 2. Identify Peaks and Troughs
    max_idx = np.argmax(future_preds)
    peak_val = future_preds[max_idx]
    peak_date = future_dates[max_idx]
    
    min_idx = np.argmin(future_preds)
    trough_val = future_preds[min_idx]
    trough_date = future_dates[min_idx]

    # 3. Volatility Analysis
    volatility = np.std(future_preds) / avg_fut if avg_fut > 0 else 0

    category_icons = {
        "Revenue": "📈",
        "Inventory": "📦",
        "Marketing": "🎯",
        "Operations": "⚙️",
        "Pricing": "💰",
        "Risk": "🛡️"
    }
    
    category_colors = {
        "Revenue": "#28a745",
        "Inventory": "#ffc107",
        "Marketing": "#6f42c1",
        "Operations": "#17a2b8",
        "Pricing": "#e83e8c",
        "Risk": "#6c757d"
    }
    
    def get_category_icon(category):
        return category_icons.get(category, "")
    
    def get_category_color(category):
        return category_colors.get(category, "#ff4b4b")
    
    def get_tooltip(category, key=None):
        tooltips = {
            "Revenue": {
                "high_growth": "• Increase ad budget 15-20%\n• Launch influencer partnerships\n• Expand product lines",
                "moderate_growth": "• Gradual ad spend increase (10%)\n• Optimize conversion funnels\n• Prepare 15% more inventory",
                "high_decline": "• Flash sales (20-30% off)\n• Loyalty rewards program\n• Email campaigns",
                "moderate_decline": "• Limited-time offers\n• Customer retention focus\n• Exclusive deals for loyal customers",
                "stable": "• Customer retention focus\n• Operational cost optimization\n• Incremental upselling"
            },
            "Inventory": "• Order 25% extra stock 2 weeks ahead\n• Ensure popular SKUs stocked\n• Coordinate priority shipping",
            "Marketing": "• Run ads 48-72hrs before peak\n• Focus on high-intent audiences\n• Schedule email promotions",
            "Operations": {
                "high_vol": "• Flexible staffing needed\n• Use part-time for peaks\n• Cross-train employees",
                "low_vol": "• Optimize full-time schedules\n• Minimize overtime costs\n• Performance incentives"
            },
            "Pricing": {
                "high_demand": "• Value-based pricing\n• Premium product tiers\n• Reduce discounts",
                "low_demand": "• Tiered discounts\n• Loyalty-based pricing\n• Maintain market share",
                "stable": "• Value-added services\n• Occasional promotions\n• Monitor competitors"
            },
            "Risk": {
                "high_risk": "• Diversify portfolio\n• 3-month cash reserves\n• Backup suppliers ready",
                "low_risk": "• Regular monitoring\n• Maintain safety stock\n• Quarterly risk review"
            }
        }
        val = tooltips.get(category, "")
        if isinstance(val, dict) and key:
            return val.get(key, "")
        return val

    # --- 1. REVENUE (Growth/Recovery/Stability) ---
    if pct_change > 10:
        recommendations.append({
            "category": "Revenue",
            "icon": get_category_icon("Revenue"),
            "color": get_category_color("Revenue"),
            "title": f"Accelerate Growth ({pct_change:.1f}% Projected)",
            "description": f"Strong upward trajectory. Increase marketing budget by 15-20%, expand product lines, and prioritize digital advertising to capitalize on momentum.",
            "tooltip": get_tooltip("Revenue", "high_growth")
        })
    elif pct_change > 5:
        recommendations.append({
            "category": "Revenue",
            "icon": get_category_icon("Revenue"),
            "color": get_category_color("Revenue"),
            "title": f"Capitalize on {pct_change:.1f}% Growth",
            "description": f"Positive trend ahead. Gradually increase ad spend by 10%, optimize conversion funnels, and prepare inventory for 15% higher demand.",
            "tooltip": get_tooltip("Revenue", "moderate_growth")
        })
    elif pct_change < -10:
        recommendations.append({
            "category": "Revenue",
            "icon": get_category_icon("Revenue"),
            "color": "#dc3545",
            "title": f"Urgent: Address {abs(pct_change):.1f}% Decline",
            "description": "Significant drop predicted. Launch flash sales (20-30% off), implement loyalty rewards, and increase social media engagement.",
            "tooltip": get_tooltip("Revenue", "high_decline")
        })
    elif pct_change < -5:
        recommendations.append({
            "category": "Revenue",
            "icon": get_category_icon("Revenue"),
            "color": "#dc3545",
            "title": f"Mitigate {abs(pct_change):.1f}% Decline",
            "description": "Sales expected to decrease. Create limited-time offers, increase customer retention efforts, and offer exclusive deals to loyal customers.",
            "tooltip": get_tooltip("Revenue", "moderate_decline")
        })
    else:
        recommendations.append({
            "category": "Revenue",
            "icon": get_category_icon("Revenue"),
            "color": get_category_color("Revenue"),
            "title": "Maintain Steady Growth",
            "description": "Stable forecast indicates consistent performance. Focus on customer retention and operational efficiency.",
            "tooltip": get_tooltip("Revenue", "stable")
        })

    # --- 2. INVENTORY (Peak Demand) ---
    days_to_peak = (peak_date - pd.Timestamp.now()).days
    if days_to_peak > 0:
        recommendations.append({
            "category": "Inventory",
            "icon": get_category_icon("Inventory"),
            "color": get_category_color("Inventory"),
            "title": f"Stock-up for Peak ({peak_date.strftime('%d %b')})",
            "description": f"Maximum demand of {peak_val:,.0f} expected. Ensure 20% safety stock is available 3-5 days before this peak.",
            "tooltip": get_tooltip("Inventory")
        })

    # --- 3. MARKETING (Ad Strategy) ---
    recommendations.append({
        "category": "Marketing",
        "icon": get_category_icon("Marketing"),
        "color": get_category_color("Marketing"),
        "title": "Optimized Ad Scheduling",
        "description": f"Run high-impact ads 48 hours before peak ({peak_date.strftime('%d %b')}). Scale back during troughs to save budget.",
        "tooltip": get_tooltip("Marketing")
    })

    # --- 4. OPERATIONS (Staffing) ---
    if volatility > 0.2:
        recommendations.append({
            "category": "Operations",
            "icon": get_category_icon("Operations"),
            "color": get_category_color("Operations"),
            "title": "Dynamic Staffing Required",
            "description": f"High volatility ({volatility:.1%}) detected. Use flexible part-time staffing to match fluctuating daily demand.",
            "tooltip": get_tooltip("Operations", "high_vol")
        })
    else:
        recommendations.append({
            "category": "Operations",
            "icon": get_category_icon("Operations"),
            "color": get_category_color("Operations"),
            "title": "Fixed Staffing Efficiency",
            "description": "Low volatility indicates predictable demand. Optimize full-time schedules to minimize overtime costs.",
            "tooltip": get_tooltip("Operations", "low_vol")
        })

    # --- 5. PRICING (Based on Demand) ---
    if pct_change > 10:
        recommendations.append({
            "category": "Pricing",
            "icon": get_category_icon("Pricing"),
            "color": get_category_color("Pricing"),
            "title": "Premium Pricing Opportunity",
            "description": "Strong demand allows for value-based pricing. Consider premium tiers and reduce discounting to maximize revenue.",
            "tooltip": get_tooltip("Pricing", "high_demand")
        })
    elif pct_change < -5:
        recommendations.append({
            "category": "Pricing",
            "icon": get_category_icon("Pricing"),
            "color": get_category_color("Pricing"),
            "title": "Competitive Pricing Response",
            "description": "Declining demand requires strategic pricing. Implement tiered discounts and loyalty-based pricing to maintain market share.",
            "tooltip": get_tooltip("Pricing", "low_demand")
        })
    else:
        recommendations.append({
            "category": "Pricing",
            "icon": get_category_icon("Pricing"),
            "color": get_category_color("Pricing"),
            "title": "Stable Pricing Strategy",
            "description": "Consistent demand supports stable pricing. Focus on value-added services and occasional promotions.",
            "tooltip": get_tooltip("Pricing", "stable")
        })

    # --- 6. RISK MANAGEMENT ---
    if volatility > 0.2 or abs(pct_change) > 10:
        recommendations.append({
            "category": "Risk",
            "icon": get_category_icon("Risk"),
            "color": get_category_color("Risk"),
            "title": "Implement Risk Mitigation",
            "description": "High-variance forecast detected. Diversify product portfolio, maintain 3-month cash reserves, and establish backup suppliers.",
            "tooltip": get_tooltip("Risk", "high_risk")
        })
    else:
        recommendations.append({
            "category": "Risk",
            "icon": get_category_icon("Risk"),
            "color": get_category_color("Risk"),
            "title": "Maintain Risk Monitoring",
            "description": "Stable forecast allows for standard risk management. Continue regular monitoring and maintain safety stock levels.",
            "tooltip": get_tooltip("Risk", "low_risk")
        })

    return recommendations
