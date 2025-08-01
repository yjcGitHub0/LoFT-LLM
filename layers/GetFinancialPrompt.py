from typing import List, Dict
import torch
import re
from collections import defaultdict
import numpy as np

def compute_uv_texts(
    uv_fundown: torch.Tensor,
    uv_stableown: torch.Tensor,
    uv_fundopt: torch.Tensor,
    uv_fundmarket: torch.Tensor,
    eps: float = 1e-6
) -> list[str]:
    uv_texts = []
    for i in range(uv_fundown.shape[0]):
        total_uv = (
            uv_fundown[i] + uv_stableown[i] +
            uv_fundopt[i] + uv_fundmarket[i] + eps
        )
        buy_ratio = uv_fundopt[i] / total_uv
        uv_texts.append(
            f"Total UV: {int(total_uv.item())}, "
            f"Selected-fund preference ratio: {buy_ratio.item():.2f}"
        )
    return uv_texts


def get_prompt(
    fund_id: str,
    dates: torch.Tensor,
    apply_amt: torch.Tensor,
    redeem_amt: torch.Tensor,
    yield_: torch.Tensor,
    uv_fundown: torch.Tensor,
    uv_stableown: torch.Tensor,
    uv_fundopt: torch.Tensor,
    uv_fundmarket: torch.Tensor,
    is_month_end: torch.Tensor,
    is_trade: torch.Tensor,
    pred_dates: torch.Tensor,
    predicted_apply: torch.Tensor,
    predicted_redeem: torch.Tensor,
    predicted_apply_high_freq: torch.Tensor,
    predicted_apply_low_freq: torch.Tensor,
    predicted_redeem_high_freq: torch.Tensor,
    predicted_redeem_low_freq: torch.Tensor,
    pred_yield_: torch.Tensor,
    pred_uv_fundown: torch.Tensor,
    pred_uv_stableown: torch.Tensor,
    pred_uv_fundopt: torch.Tensor,
    pred_uv_fundmarket: torch.Tensor,
    pred_is_month_end: torch.Tensor,
    pred_is_trade: torch.Tensor,
    pred_len: int
) -> str:
    uv_texts = compute_uv_texts(uv_fundown, uv_stableown, uv_fundopt, uv_fundmarket)
    pred_uv_texts = compute_uv_texts(pred_uv_fundown, pred_uv_stableown, pred_uv_fundopt, pred_uv_fundmarket)

    prompt = f"""### Role: Financial Sequence Analyst

You are a quantitative analyst focusing on mutual fund capital flow modeling.
Your task is to **refine the predicted daily subscription (`apply_amt`) and redemption (`redeem_amt`) values for the next {pred_len} calendar days** for mutual fund **{fund_id}**, leveraging historical transaction behavior, macroeconomic indicators, and behavioral signals.

---

### Feature Description

- `apply_amt`: Net fund subscriptions (actual values).
- `redeem_amt`: Net fund redemptions (actual values).
- `yield`: 1-year interbank CD yield (%), reflecting opportunity cost.
- `is_trade`: 1 = trading day; 0 = non-trading day.
- `is_month_end`: 1 = last trading day of the month, often tied to institutional behavior.
- `Total UV`: Total user views on fund-related content (attention measure).
- `Selected-fund preference ratio`: Buy-side intent signal = `uv_fundopt / total_uv`.

---

### Historical Trading Days

Recent observations for fund `{fund_id}`:
"""
    for i in range(len(dates)):
        if is_trade[i].item() == 0:
            continue
        prompt += (
            f"- {dates[i]} | apply: {apply_amt[i].item():.3f}, redeem: {redeem_amt[i].item():.3f}, "
            f"yield: {yield_[i].item():.6f}, {uv_texts[i]}, month_end: {is_month_end[i].item()}\n"
        )

    prompt += f"""

---

### Prediction Window: Next {pred_len} Calendar Days

The following values are **model-generated forecasts** of future flows. Please treat them as preliminary estimates subject to refinement:
"""
    for i in range(pred_len):
        prompt += (
            f"- {pred_dates[i]} | predicted apply: {predicted_apply[i].item():.3f}, "
            f"predicted redeem: {predicted_redeem[i].item():.3f}, "
            f"yield: {pred_yield_[i].item():.6f}, {pred_uv_texts[i]}, "
            f"month_end: {pred_is_month_end[i].item()}, is_trade: {pred_is_trade[i].item()}\n"
        )

    prompt += f"""

---

### Frequency Component Reference (Optional)

For each predicted value, the model also provides two internal signal components:

- `predicted_apply = low_freq + high_freq`
- `predicted_redeem = low_freq + high_freq`

These components are **not actual cash flows**, but help explain the forecast's structure.
If this section affects your judgment, you may **ignore** its contents.

Apply-side:
- low_freq: [{', '.join(f'{v.item():.3f}' for v in predicted_apply_low_freq)}]
- high_freq: [{', '.join(f'{v.item():.3f}' for v in predicted_apply_high_freq)}]

Redeem-side:
- low_freq: [{', '.join(f'{v.item():.3f}' for v in predicted_redeem_low_freq)}]
- high_freq: [{', '.join(f'{v.item():.3f}' for v in predicted_redeem_high_freq)}]
"""

    prompt += f"""
    
---

### Instructions

Please adjust the model-generated `apply_amt` and `redeem_amt` predictions based on:

- Investor intent signals (UVs and preference ratio);
- Yield as an indicator of fund attractiveness;
- Structural seasonality (month-end);
- Historical trends and financial intuition.

🧠 Treat the provided forecast as a **base estimate** requiring refinement. Pay attention to:
- If `is_trade == 0`, both apply and redeem **must** be 0.0.
- Higher **preference ratio** → stronger buy-side pressure.
- Higher **yield** → increased redemption potential.

---

### Output Format (Strict)

Return **two lists of {pred_len} float values**, representing:

1. Refined `apply_amt` predictions
2. Refined `redeem_amt` predictions

⚠️ No explanations, comments, or variable names.

Example:

[{', '.join(['float']*pred_len)}]
[{', '.join(['float']*pred_len)}]
"""
    return prompt



