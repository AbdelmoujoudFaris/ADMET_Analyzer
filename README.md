# 🔬 ADMET Analyzer

> Publication-quality ADMET profiling tool for drug discovery with 2D molecular structure visualization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![RDKit](https://img.shields.io/badge/RDKit-2022+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

---

## 📋 Overview

**ADMET Analyzer** is a desktop GUI application for rapid ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) profiling of drug candidates. Built to Nature/Science publication standards, it generates high-resolution figures suitable for direct use in scientific manuscripts.

### Key Features

- **Lipinski's Rule of Five** — oral drug-likeness filter
- **Veber's Rule** — oral bioavailability prediction
- **Ghose Filter** — drug-likeness assessment
- **Full ADMET Predictions** — BBB penetration, CYP450, renal clearance, hepatotoxicity
- **QED Scoring** — quantitative estimate of drug-likeness
- **2D Molecular Structures** — embedded directly in scatter plots
- **Batch Analysis** — process hundreds of compounds from CSV/SDF/TXT
- **Publication-Quality Figures** — exported at 300 DPI

---

## 🖥️ Screenshots

| Structures Grid | MW vs LogP | QED Ranking |
|---|---|---|
| 2D structures with properties | Lipinski space + embedded structures | Color-coded quality ranking |

---

## ⚙️ Installation

```bash
pip install rdkit pillow pandas numpy matplotlib seaborn
```

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/admet-analyzer.git
cd admet-analyzer
```

---

## 🚀 Usage

```bash
python admet_analyzer.py
```

### Input Formats

**Option 1 — Named compounds (recommended):**
```
Aspirin,CC(=O)Oc1ccccc1C(=O)O
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O
```

**Option 2 — SMILES only (auto-named A, B, C...):**
```
CC(=O)Oc1ccccc1C(=O)O
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

**Option 3 — Load from file:**
- `.csv` with `SMILES` column (optional `NAME`/`ID` column)
- `.sdf` standard structure file
- `.txt` one SMILES or `NAME,SMILES` per line

---

## 📊 Output Visualizations

| Tab | Description |
|-----|-------------|
| 🔬 Structures | 2D grid of molecular structures with QED/MW/LogP |
| 📐 Drug Space | MW vs LogP and TPSA vs Rotatable Bonds with embedded structures |
| 📊 QED Quality | Score distribution histogram + compound ranking |
| 📈 Properties | Boxplots + parallel coordinates multi-property comparison |
| 🔥 Heatmap | Normalized property heatmap for compound comparison |

---

## 📤 Export

Click **💾 Export Results** to save:
- `results.csv` — full ADMET data table
- `results_report.txt` — text summary
- `results_*.png` — all figures at 300 DPI

---

## 🧪 Calculated Properties

| Property | Method | Reference |
|----------|--------|-----------|
| MW, LogP, HBD, HBA | RDKit Descriptors | Lipinski et al. 2001 |
| TPSA, RotBonds | RDKit Descriptors | Veber et al. 2002 |
| QED Score | RDKit QED module | Bickerton et al. 2012 |
| BBB Penetration | LogP-TPSA-HBD model | Estimated |
| CYP450 Interaction | LogP + aromatic rings | Estimated |
| Ames Mutagenicity | Structural alerts | Estimated |

> ⚠️ **Disclaimer:** ADMET predictions are computational estimates based on physicochemical rules and structural alerts. They are intended for early-stage prioritization only and do not replace experimental assays.

---

## 📚 References

- Lipinski et al. *Adv Drug Deliv Rev* 2001;46:3–26
- Veber et al. *J Med Chem* 2002;45:2615–23
- Bickerton et al. *Nat Chem* 2012;4:90–98
- Daina et al. *Sci Rep* 2017;7:42717 (SwissADME)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🤝 Contributing

Pull requests welcome. For major changes, please open an issue first to discuss what you would like to change.

```
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```
