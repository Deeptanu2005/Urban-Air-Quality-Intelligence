# 🌦️ Urban Air Quality Intelligence

> **Explore. Compare. Diagnose. Understand.**

An interactive Streamlit dashboard for analysing urban air quality across **26 Indian cities** using CPCB city-day measurements from **2015–2020**.

---

## 📊 At a Glance

| 🏙️ Cities | 📍 Observations | 📅 Period | 🧪 Pollutants |
|:---:|:---:|:---:|:---:|
| **26** | **29,531** | **2015–2020** | **12** |

<br>

| Analysis | Visualisation | Statistics | Interface |
|---|---|---|---|
| AQI patterns | Interactive Plotly | Pearson r | Dynamic filters |
| City comparisons | Heatmaps | Mann–Whitney U | Dark UI |
| Pollution analysis | Distributions | p-values | Responsive layout |
| AQI spikes | Correlation plots | Rolling z-score | Live KPIs |

---

## ✦ Dashboard Structure

```text
Sidebar
│
├── Region presets
├── City selection
├── Year range
├── Season filter
├── AQI category filter
└── Live selection summary
        │
        ▼
Main Dashboard
│
├── Hero overview
├── KPI cards
├── Supporting statistics
│
└── Analytical Tabs
    │
    ├── 📈 AQI Intelligence
    ├── 🏙️ City Intelligence
    ├── 🧪 Pollution Intelligence
    ├── 🔬 Diagnostic Intelligence
    └── 🗄️ Data Quality
```

### 🎛️ ANALYSIS CONTROLS (Filters)

**Region** · **City** · **Year** · **Season** · **AQI Category**

All KPIs and visualisations respond dynamically to the current selection.

---

## 🧠 Analytics

| Area | Methods |
|---|---|
| **AQI** | Distribution · categories · yearly/monthly/seasonal trends |
| **Cities** | Rankings · city × year comparison · heatmaps |
| **Pollutants** | Distributions · concentration profiles · AQI relationships |
| **Correlation** | Pearson correlation · 13×13 matrix · per-pollutant p-values (AQI pairs) |
| **Risk Analysis** | AQI ≤ 100 vs AQI > 200 · Mann–Whitney U |
| **Spike Detection** | 30-day rolling z-score · threshold > 2.0 |
| **Driver Analysis** | Normalised pollutant contribution by city, year & season |

> **Statistical note:** Results describe observed associations in the dataset; correlation does not establish causation.

---

## 📌 Dynamic KPIs

| KPI | Definition |
|---|---|
| **Average AQI** | Mean AQI across filtered records |
| **Pollution Risk Rate** | % of days with AQI > 200 |
| **Dominant Pollution Driver** | Highest normalised mean pollutant concentration |

**Normalised concentration**

```text
mean pollutant concentration ÷ reference upper bound
```

The dominant driver is recalculated dynamically according to the active filters.

---

## 🧹 Data Preparation

| Data issue | Treatment |
|---|---|
| Missing AQI | City-month median → city median |
| Missing pollutants | City-month → city → global median |
| Zero Benzene / Toluene / Xylene / CO | Treated as below-detection-limit |
| AQI > 999 | Capped at 999 |
| Duplicate `(City, Date)` | Checked |

### Key Data Notes

| Item | Note |
|---|---|
| **AQI imputation** | Preserves some seasonal variation; sparse months may understate extremes |
| **AQI spikes** | Relative to each city's 30-day baseline |
| **CO** | Zero-value treatment can affect high-risk elevation ratios |
| **Lucknow PM10** | Global PM10 median used because source contains zero measurements |
| **Ahmedabad** | 93 AQI values above 999 capped during preprocessing |

Full cleaning statistics are available inside **Data Quality**.

---

## 🌍 Dataset

| Attribute | Value |
|---|---|
| **Source** | Central Pollution Control Board (CPCB), India |
| **Dataset** | `city_day.csv` |
| **Rows** | 29,531 |
| **Columns** | 16 |
| **Cities** | 26 |
| **Coverage** | 2015-01-01 → 2020-07-01 |

**Pollutants**

`PM2.5` · `PM10` · `NO` · `NO₂` · `NOₓ` · `NH₃` · `CO` · `SO₂` · `O₃` · `Benzene` · `Toluene` · `Xylene`

### Download

🔗 **[Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india?select=city_day.csv)**
or,  
🔗 **[Google Drive](https://drive.google.com/file/d/1ADfDemnRyd1lutyx1u0srnDBfsj1dKJz/view?usp=sharing)**

---

## ⚙️ Data Flow

```text
city_day.csv
     │
     ▼
Data Validation
     │
     ▼
Cleaning & Imputation
     │
     ▼
Feature Engineering
     │
     ▼
Filtered Dataset
     │
     ├── AQI Analysis
     ├── City Analysis
     ├── Pollution Analysis
     └── Diagnostic Analysis
```

---

## 🛠️ Built With

| Technology | Role |
|---|---|
| 🐍 `Python` | Application & analysis |
| 🎈 `Streamlit` | Dashboard |
| 🐼 `Pandas` | Data processing |
| 🔢 `NumPy` | Numerical operations |
| 📊 `Plotly` | Interactive charts |
| 📐 `SciPy` | Statistical analysis |

---

## 📁 Project Structure

```text
.
├── DeeptanuSen_UrbanAirQualityIntelligence.py  # Streamlit application
├── city_day.csv        # Dataset
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🚀 Run Locally

### Prerequisites

- Python 3.11+
- Git
- `city_day.csv` *(optional when using the built-in dataset fallback)*

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <project-directory>
```

### 2. Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run DeeptanuSen_UrbanAirQualityIntelligence.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📦 Dataset Loading

The application uses a simple dataset-loading pipeline:

```text

city_day.csv exists locally
        │
        ├── YES → Load local dataset
        │
        └── NO
             │
             ▼
       Streamlit cache
             │
             ├── HIT  → Load cached dataset
             │
             └── MISS
                    │
                    ▼
             Google Drive
                    │
                    ▼
             Save local copy
                    │
                    ▼
             Load dataset
```

This allows the application to start without requiring the dataset to be manually placed in the project directory when the Google Drive fallback is available.

---

## ✨ Highlights

| | |
|---|---|
| 📊 **29,531** observations | 🏙️ **26** cities |
| 📅 **2015–2020** | 🧪 **12** pollutants |
| 🎛️ Dynamic filtering | 📈 Interactive visualisations |
| 🔬 Statistical diagnostics | 🧹 Integrated data-quality analysis |

---

## 👤 Author

**Deeptanu Sen**

Developed as part of the **IBM SkillsBuild Data Analytics with AI Internship 2026**.

> Making air-quality data something you can explore — not just read.

---

### 📚 References

- [CPCB India](https://cpcb.nic.in/)
- [Kaggle - Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- [Dataset - Google Drive](https://drive.google.com/file/d/1ADfDemnRyd1lutyx1u0srnDBfsj1dKJz/view?usp=sharing)