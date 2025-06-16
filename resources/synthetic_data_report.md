**Market Mix Modelling**

The main goal of Market Mix Modelling is to understand how different marketing and advertising efforts affect sales. We want to find out how many units are sold because of each type of advertisement. This helps in figuring out how to spend the marketing budget in a smarter way.

The project is divided into four main parts:

1. **Exploratory Data Analysis** – to explore and understand the data
2. **Modelling** – to build a model that links ads to sales
3. **Contribution Calculation** – to see how much each marketing activity adds to sales
4. **Business Impact Analysis and Suggestions** – to give useful recommendations based on what we learn

To test the whole process properly, we are using a **top-down approach**. This means we start by setting the expected results, such as the effect of each ad type (coefficients) and their contributions to sales. Then we create synthetic (fake but realistic) data based on those inputs. This helps us check if the model gives the correct results or not.

To make sure the code is flexible and easy to scale later, we use a **configuration dictionary**. This dictionary holds all the key settings and values we need for generating the data:

```python
data_gen_params = {
    # Types of products and their average prices
    "products": {"premium": 80, "mid_tier": 30, "low_tier": 20},

    # Start and end dates for the data
    "min_date": pd.to_datetime("2023-01-01"),
    "max_date": pd.to_datetime("2025-01-01"),

    # Key performance indicators (KPIs) and their cost per click (CPC) or weight
    "kpis": {
        "product_level": {"branded": 0.3, "nonbranded": 0.2},
        "brand_level": {"insta": 2, "fb": 3},
    },

    # Important dates related to events
    "events": [
        "2024-05-13",  # Product launch
        "2025-09-14",  # Prime Day
    ],

    # Coefficients that show the impact of each KPI on sales for each product type
    "kpi_coefs": {
        "product_level": {
            "branded": {"premium": 15, "mid_tier": 10, "low_tier": 8},
            "nonbranded": {"premium": 12, "mid_tier": 9, "low_tier": 7},
            "price": {"premium": -0.5, "mid_tier": -0.4, "low_tier": -0.3},
            "oos": {"premium": -1.2, "mid_tier": -0.8, "low_tier": -0.5},  # out-of-stock impact
        },
        "brand_level": {"insta": 2.5, "fb": 2.0},
    },
}
```

To support the generation of synthetic data for this exercise, we use a configuration dictionary called `data_gen_params`. This dictionary acts as the central control system for simulating data in a way that mimics real-world marketing and sales behavior.

For this exercise, we have picked **three types of products**:

* A **premium product** priced at 80
* A **mid-tier product** priced at 30
* A **low-tier product** priced at 20

These different price levels help us see how marketing affects products in different segments. 

Next, we set a **date range** for the data. For this project, we simulate two years of activity, from January 1, 2023, to January 1, 2025. This gives us enough time to include different marketing cycles, seasonal effects, and major events.

We also define key **marketing performance indicators (KPIs)**. These are split into two levels:

* **Product-level KPIs**, such as branded and non-branded ads, which directly influence individual products.
* **Brand-level KPIs**, like Instagram and Facebook campaigns, which support the overall brand and may impact multiple products indirectly.

Special **events** are included to model real-world spikes in activity, such as a **product launch** or **Prime Day**. These help us test if the model can detect and account for unexpected changes in sales due to non-marketing factors.

One of the most critical parts of this setup is the list of **coefficients**. These values tell us how much each factor (like an ad type or a price change) is expected to impact sales. For example, a high coefficient for branded ads on premium products means those ads are very effective. Negative coefficients, like those for pricing or out of stocks, indicate a drop in sales when those conditions occur.

By using this structured configuration, we generate synthetic data that behaves in a realistic and controlled way. 

If the model can match the predefined coefficients and contributions, it gives us confidence that the approach works—and can later be applied to real data for actual business decisions.


In Market Mix Modelling (MMM), **coefficients** are the numbers that tell us how much each marketing activity impacts sales. They represent the **incremental units** sold for each unit change in the marketing input, all else held constant.

---

### 📈 What the Coefficients Mean

1. **Magnitude (Size)**
   A larger coefficient means a bigger impact on sales.
   *Example:* A coefficient of **15** for premium‑brand ads means +1 unit of that ad gives +15 sales units—showing strong effectiveness.

2. **Direction (Positive or Negative)**

   * **Positive** means more ad spend → more sales.
   * **Negative** means that factor **reduces** sales (e.g., higher price reduces demand).
     This aligns with real-world expectations—price hikes or out-of-stock issues should reduce sales.

3. **Real‑world realism**
   These values are based on typical marketing behavior:

   * **Branded ads** often have higher impact than **non-branded** because they reinforce brand loyalty ([vijayendra-dwari.medium.com][1], [reddit.com][2]).
   * **Price sensitivity**: expensive products are more sensitive to price changes, so a coefficient of −0.5 for premium vs –0.3 for low-tier mimics buyer behavior .

---

### Why These Specific Values?

* **Branded Ads** Premium (15) > Branded Mid (10) > Branded Low (8)
  Premium customers expect more from brand messaging—they respond strongly, so we give a higher coefficient.

* **Non-Branded Ads (12 / 9 / 7)**
  Still valuable, but less impactful since they lack brand context. That’s why the numbers are slightly lower.

* **Price (–0.5 / –0.4 / –0.3)**
  Reflects price elasticity: higher-priced items suffer more when price increases.

* **Out-of-Stock (–1.2 / –0.8 / –0.5)**
  If a product is unavailable, sales drop sharply—especially for premium items where customers expect consistency.

* **Brand-level ads (2.5 for Instagram, 2.0 for Facebook)**
  These contribute to general brand awareness rather than specific products and thus carry moderate positive influence.

These coefficients simulate **real-world marketing behavior** by incorporating known effects like **adstock**, **diminishing returns**, and **elasticity** ([getrecast.com][3]).

---

### Why It Matters

* **Simulating Real Scenarios**
  By choosing realistic values, we ensure our synthetic data behaves like real-world data—ads saturate, prices matter, stockouts hurt sales.

* **Model Validation**
  When we train our MMM model, we expect it to recover similar coefficient values. If it does, we know the model is **working correctly**.

* **Planning & Optimization**
  These insights help marketers simulate what-if scenarios:

  * Reallocate budget across channels
  * Understand how raising price might impact volume
  * Plan around events and inventory issues

---




[1]: https://vijayendra-dwari.medium.com/understanding-market-mix-modeling-a-comprehensive-guide-2a5bfb7be4a6?utm_source=chatgpt.com "Understanding Market Mix Modeling: A Comprehensive Guide | by Vijayendra Dwari | Geek Culture | Medium"
[2]: https://www.reddit.com/r/PPC/comments/1bikpob?utm_source=chatgpt.com "Google Ads: Brand vs Non-brand CPL ratio"
[3]: https://getrecast.com/scenario-planning/?utm_source=chatgpt.com "Marketing Mix Modeling for Scenario Planning - Recast"
[4]: https://mass-analytics.com/marketing-mix-modeling-blogs/regression-analysis-for-marketing-mix-modeling/?utm_source=chatgpt.com "Regression Analysis for Marketing Mix Modeling - MASS Analytics"
[5]: https://www.reddit.com/r/PPC/comments/1iogsgi?utm_source=chatgpt.com "MMM in Practice: living up to the hype?"
[6]: https://www.reddit.com/r/datascience/comments/13l7aek?utm_source=chatgpt.com "Marketing Mix Model - ROI and budget allocation"
[7]: https://www.listendata.com/2019/09/marketing-mix-modeling.html?utm_source=chatgpt.com "A Complete Guide to Marketing Mix Modeling"
