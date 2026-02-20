"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ADMET Analyzer Pro — Publication-Quality Drug Discovery Tool      ║
║                                                                              ║
║  Computes 35+ pharmacokinetic & physicochemical descriptors from SMILES.     ║
║  Implements Lipinski Ro5, Veber, Ghose, REOS, and Lead-like filters.         ║
║  Generates 2D molecular depictions, scatter/box/heatmap/parallel plots.      ║
║                                                                              ║
║  References:                                                                 ║
║    • Lipinski et al.  Adv Drug Deliv Rev 2001;46:3-26                        ║
║    • Veber et al.     J Med Chem 2002;45:2615-23                             ║
║    • Egan et al.      J Med Chem 2000;43:3867-77                             ║
║    • Ghose et al.     J Comb Chem 1999;1:55-68                               ║
║    • Bickerton et al. Nat Chem 2012;4:90-8 (QED)                             ║
║    • Daina et al.     Sci Rep 2017;7:42717 (SwissADME)                       ║
║                                                                              ║
║  Source: https://github.com/AbdelmoujoudFaris/ADMET_Analyzer                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import io, base64, sys
from datetime import datetime
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── RDKit ────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import (
        Descriptors, Crippen, Lipinski, AllChem,
        rdMolDescriptors, QED, Draw, FilterCatalog
    )
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem.FilterCatalog import FilterCatalogParams
    from PIL import Image as PILImage   # for 2D depiction embedding
    RDKIT_AVAILABLE = True
    PIL_AVAILABLE   = True
except ImportError:
    RDKIT_AVAILABLE = False
    PIL_AVAILABLE   = False
    print("WARNING: RDKit not available.  pip install rdkit pillow")

# ─────────────────────────────────────────────────────────────────────────────
#  Colour palette  (Nature / Science style)
# ─────────────────────────────────────────────────────────────────────────────
PAL = {
    "blue":   "#2E86AB",
    "green":  "#27AE60",
    "red":    "#E74C3C",
    "orange": "#F39C12",
    "purple": "#8E44AD",
    "teal":   "#1ABC9C",
    "grey":   "#95A5A6",
    "navy":   "#2C3E50",
    "bg":     "#F7F9FC",
    "panel":  "#FFFFFF",
}
PASS_COLOR = PAL["green"]
FAIL_COLOR = PAL["red"]


# ══════════════════════════════════════════════════════════════════════════════
#  ADMETCalculator  –  35+ descriptors
# ══════════════════════════════════════════════════════════════════════════════
class ADMETCalculator:
    """
    All descriptors are computed directly from SMILES via RDKit.
    Methods are static so the class can be used without instantiation.
    """

    # ── Drug-likeness filters ─────────────────────────────────────────────────

    @staticmethod
    def lipinski(mol):
        """Lipinski Rule-of-Five  (Adv Drug Deliv Rev 2001)"""
        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd  = Lipinski.NumHDonors(mol)
        hba  = Lipinski.NumHAcceptors(mol)
        v    = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        return dict(MW=mw, LogP=logp, HBD=hbd, HBA=hba,
                    Lipinski_Violations=v, Lipinski_Pass=v <= 1)

    @staticmethod
    def veber(mol):
        """Veber Rule for oral bioavailability  (J Med Chem 2002)"""
        rb   = Lipinski.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        return dict(RotatableBonds=rb, TPSA=tpsa,
                    Veber_Pass=(rb <= 10) and (tpsa <= 140))

    @staticmethod
    def ghose(mol):
        """Ghose drug-likeness filter  (J Comb Chem 1999)"""
        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        na   = mol.GetNumAtoms()
        mr   = Crippen.MolMR(mol)
        ok   = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6) and \
               (20 <= na <= 70) and (40 <= mr <= 130)
        return dict(Ghose_Pass=ok, MolRefractivity=mr, NumAtoms=na)

    @staticmethod
    def reos(mol):
        """Rapid Elimination Of Swill  (Drug Discov Today 2002)"""
        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd  = Lipinski.NumHDonors(mol)
        hba  = Lipinski.NumHAcceptors(mol)
        charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        rb   = Lipinski.NumRotatableBonds(mol)
        na   = mol.GetNumHeavyAtoms()
        ok   = (200 <= mw <= 500) and (-5 <= logp <= 5) and \
               (0 <= hbd <= 5) and (0 <= hba <= 10) and \
               (-2 <= charge <= 2) and (rb <= 8) and (15 <= na <= 50)
        return dict(REOS_Pass=ok, HeavyAtoms=na, FormalCharge=charge)

    @staticmethod
    def lead_like(mol):
        """Lead-like filter  (Drug Discov Today 2003)"""
        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rb   = Lipinski.NumRotatableBonds(mol)
        ok   = (mw <= 350) and (logp <= 3) and (rb <= 7)
        return dict(LeadLike_Pass=ok)

    @staticmethod
    def egan(mol):
        """Egan egg model  (J Med Chem 2000)"""
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        ok   = (logp <= 5.88) and (tpsa <= 131.6)
        return dict(Egan_Pass=ok)

    # ── Physicochemical extras ────────────────────────────────────────────────

    @staticmethod
    def physicochemical(mol):
        """Extended physicochemical properties"""
        mw       = Descriptors.MolWt(mol)
        logp     = Crippen.MolLogP(mol)
        tpsa     = Descriptors.TPSA(mol)
        rings    = rdMolDescriptors.CalcNumRings(mol)
        aro_r    = Lipinski.NumAromaticRings(mol)
        ali_r    = Lipinski.NumAliphaticRings(mol)
        sp3_frac = rdMolDescriptors.CalcFractionCSP3(mol)
        stereo   = len(rdMolDescriptors.CalcChiralCenters(mol, includeUnassigned=True))
        alogd    = logp - 1.0 if tpsa > 70 else logp          # very rough LogD proxy
        return dict(
            NumRings=rings, NumAromaticRings=aro_r,
            NumAliphaticRings=ali_r, Fsp3=round(sp3_frac, 3),
            StereoCenters=stereo, LogD_est=round(alogd, 2),
            MolFormula=rdMolDescriptors.CalcMolFormula(mol)
        )

    # ── ADMET properties ──────────────────────────────────────────────────────

    @staticmethod
    def absorption(mol):
        """Absorption endpoints"""
        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd  = Lipinski.NumHDonors(mol)
        hba  = Lipinski.NumHAcceptors(mol)

        # Human Intestinal Absorption probability (rule-based)
        hia = 95 if tpsa < 60 else 75 if tpsa < 140 else 30

        # Caco-2 permeability
        caco2 = "High" if (logp > 1 and tpsa < 90) else \
                "Medium" if tpsa < 140 else "Low"

        # Pgp substrate heuristic
        pgp_sub = "Yes" if (mw > 400 and tpsa > 60 and hbd > 3) else "No"

        # Bioavailability score (F20/F30 Egan)
        f20 = (logp <= 5.88) and (tpsa <= 131.6)
        f30 = (logp <= 5.88) and (tpsa <= 131.6) and (mw <= 500)

        # Water solubility estimate (ESOL-like)
        rb   = Lipinski.NumRotatableBonds(mol)
        aro  = Lipinski.NumAromaticRings(mol)
        esol = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*aro
        sol  = "High" if esol > -2 else "Medium" if esol > -4 else "Low"

        lip  = ADMETCalculator.lipinski(mol)
        veb  = ADMETCalculator.veber(mol)
        oral = lip["Lipinski_Pass"] and veb["Veber_Pass"]

        return dict(
            HIA_Probability=hia, Caco2_Permeability=caco2,
            Pgp_Substrate=pgp_sub, Oral_Bioavailability=oral,
            BioAvailability_F20=f20, BioAvailability_F30=f30,
            ESOL_LogS=round(esol, 2), WaterSolubility=sol
        )

    @staticmethod
    def distribution(mol):
        """Distribution endpoints"""
        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd  = Lipinski.NumHDonors(mol)

        # BBB score (Ertl)
        bbb_score = logp - (tpsa / 100) - hbd
        bbb_cat   = "High" if bbb_score > 0.5 else \
                    "Medium" if bbb_score > -0.5 else "Low"

        # Plasma protein binding (crude linear model)
        ppb = min(100, max(0, 80 + 10*(logp - 2)))

        # VD heuristic
        vd  = "High (>1 L/kg)" if logp > 3 else \
              "Medium" if logp > 1 else "Low (<0.5 L/kg)"

        # CNS penetration (MPO ≥ 4)
        mw_score  = 1 if mw < 360 else 0.5 if mw < 500 else 0
        pka_proxy = 1  # cannot compute without pKa tool
        mpo_score = mw_score + pka_proxy + (1 if logp < 3 else 0.5) + \
                    (1 if tpsa < 90 else 0.5) + (1 if hbd <= 3 else 0)
        cns_mpo   = "Favorable (≥4)" if mpo_score >= 4 else \
                    "Moderate (2-4)" if mpo_score >= 2 else "Poor"

        return dict(
            BBB_Penetration=bbb_cat, BBB_Score=round(bbb_score, 3),
            PlasmaProteinBinding_pct=round(ppb, 1),
            VolumeDistribution=vd, CNS_MPO=cns_mpo
        )

    @staticmethod
    def metabolism(mol):
        """CYP450 interaction predictions"""
        logp = Crippen.MolLogP(mol)
        mw   = Descriptors.MolWt(mol)
        aro  = Lipinski.NumAromaticRings(mol)
        hba  = Lipinski.NumHAcceptors(mol)

        sub3a4  = logp > 2 and aro >= 1
        sub2d6  = aro >= 1 and any(a.GetAtomicNum() == 7 for a in mol.GetAtoms())
        sub2c9  = hba >= 2 and logp > 1
        inh3a4  = logp > 3 and aro >= 2
        inh2d6  = aro >= 2 and mw < 450
        met_stab = "Low" if logp > 3 and aro >= 1 else \
                   "Medium" if logp > 1 else "High"

        return dict(
            CYP3A4_Substrate=sub3a4, CYP2D6_Substrate=sub2d6,
            CYP2C9_Substrate=sub2c9, CYP3A4_Inhibitor=inh3a4,
            CYP2D6_Inhibitor=inh2d6, MetabolicStability=met_stab
        )

    @staticmethod
    def excretion(mol):
        """Renal / biliary clearance"""
        mw   = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        if mw < 300 and logp < 2:
            cl = "High"
        elif mw < 500 and logp < 4:
            cl = "Medium"
        else:
            cl = "Low"

        t12 = "Short (<2 h)" if mw < 300 else \
              "Medium (2-8 h)" if mw < 500 else "Long (>8 h)"

        biliary = "Likely" if mw > 500 and logp > 2 else "Unlikely"

        return dict(
            RenalClearance=cl, HalfLife_Category=t12,
            BiliaryClearance=biliary
        )

    @staticmethod
    def toxicity(mol):
        """Toxicity alerts — PAINS, structural flags, Ames, hERG"""
        smiles  = Chem.MolToSmiles(mol)
        logp    = Crippen.MolLogP(mol)
        hba     = Lipinski.NumHAcceptors(mol)
        mw      = Descriptors.MolWt(mol)
        aro     = Lipinski.NumAromaticRings(mol)

        # Structural alerts
        alerts = []
        _checks = [
            ('[N+](=O)[O-]',        'Nitro_group'),
            ('c1ccccc1Cl',          'Halogenated_arene'),
            ('c1ccccc1Br',          'Halogenated_arene'),
            ('[$(NC(=O))]',         'Amide_alert'),
            ('C(=S)',               'Thione'),
            ('[SH]',                'Thiol'),
            ('N=N',                 'Azo_group'),
            ('[N]=[N+]=[N-]',       'Azide'),
            ('c1ccc2c(c1)cccc2',    'Naphthalene_core'),
        ]
        for pat, label in _checks:
            try:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pat)):
                    if label not in alerts:
                        alerts.append(label)
            except Exception:
                pass

        # Aromatic amine → Ames positive signal
        aro_amine = mol.HasSubstructMatch(Chem.MolFromSmarts('Nc1ccccc1'))
        ames      = aro_amine or len(alerts) > 0

        # Hepatotoxicity risk
        hepato = "High" if (logp > 5 or hba > 12 or mw > 600) else "Low"

        # hERG blocker proxy
        herg = "Risk" if (logp > 3 and aro >= 2 and mw > 300) else "Low"

        # Acute oral toxicity Cramer class (rule-of-thumb)
        cramer = "III (high)" if len(alerts) >= 2 else \
                 "II (medium)" if len(alerts) == 1 else "I (low)"

        # PAINS via RDKit FilterCatalog
        pains_alerts = []
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            catalog = FilterCatalog.FilterCatalog(params)
            for entry in catalog.GetMatches(mol):
                pains_alerts.append(entry.GetDescription())
        except Exception:
            pass

        return dict(
            Toxicity_Alerts=alerts if alerts else ["None"],
            PAINS_Alert=len(pains_alerts) > 0,
            PAINS_Descriptions=pains_alerts if pains_alerts else ["None"],
            Ames_Mutagenicity=ames,
            Hepatotoxicity_Risk=hepato,
            hERG_Risk=herg,
            Cramer_Class=cramer
        )

    @staticmethod
    def drug_likeness(mol):
        """QED + Synthetic Accessibility (approximation)"""
        try:
            qed_score = QED.qed(mol)
        except Exception:
            qed_score = 0.5

        mw  = Descriptors.MolWt(mol)
        rb  = Lipinski.NumRotatableBonds(mol)
        r   = Lipinski.NumAliphaticRings(mol) + Lipinski.NumAromaticRings(mol)
        sa  = min(10, max(1, 1 + (mw / 100) + (rb / 2) + r))

        # Fsp3 contribution: higher Fsp3 → better drug-likeness signal
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        dl   = "High" if qed_score > 0.67 else \
               "Medium" if qed_score > 0.34 else "Low"

        return dict(
            QED_Score=round(qed_score, 4),
            SyntheticAccessibility=round(sa, 2),
            DrugLikeness=dl, Fsp3=round(fsp3, 3)
        )

    # ── Master function ───────────────────────────────────────────────────────

    @staticmethod
    def full_profile(smiles_or_mol):
        """Compute all 35+ descriptors and return a flat dict."""
        if isinstance(smiles_or_mol, str):
            mol = Chem.MolFromSmiles(smiles_or_mol)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles_or_mol}")
        else:
            mol = smiles_or_mol

        profile = {}
        for fn in (
            ADMETCalculator.lipinski,
            ADMETCalculator.veber,
            ADMETCalculator.ghose,
            ADMETCalculator.reos,
            ADMETCalculator.lead_like,
            ADMETCalculator.egan,
            ADMETCalculator.physicochemical,
            ADMETCalculator.absorption,
            ADMETCalculator.distribution,
            ADMETCalculator.metabolism,
            ADMETCalculator.excretion,
            ADMETCalculator.toxicity,
            ADMETCalculator.drug_likeness,
        ):
            try:
                profile.update(fn(mol))
            except Exception as e:
                print(f"[WARN] {fn.__name__}: {e}")
        return profile


# ══════════════════════════════════════════════════════════════════════════════
#  2D Molecular Depiction
# ══════════════════════════════════════════════════════════════════════════════
class MolDepiction:
    """Generate PIL Images of 2D molecular structures using RDKit."""

    @staticmethod
    def mol_to_pil(mol, size=(300, 250)):
        """Return a PIL Image of the 2D structure."""
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().addAtomIndices       = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        bio = io.BytesIO(drawer.GetDrawingText())
        return PILImage.open(bio)

    @staticmethod
    def grid_to_pil(mols, labels=None, mols_per_row=4, sub_img_size=(250, 200)):
        """Return a PIL Image grid of multiple structures."""
        for m in mols:
            AllChem.Compute2DCoords(m)
        img = Draw.MolsToGridImage(
            mols,
            molsPerRow=mols_per_row,
            subImgSize=sub_img_size,
            legends=labels or [""] * len(mols),
            returnPNG=False,
            kekulize=True,
        )
        return img


# ══════════════════════════════════════════════════════════════════════════════
#  Publication-quality plotting
# ══════════════════════════════════════════════════════════════════════════════
class ADMETPlotter:

    @staticmethod
    def style():
        """Apply Nature/Science publication style."""
        try:
            plt.style.use("seaborn-v0_8-paper")
        except Exception:
            pass
        sns.set_palette("husl")
        plt.rcParams.update({
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.linewidth": 1.2,
            "grid.alpha": 0.3,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "white",
        })

    # ── individual plots ──────────────────────────────────────────────────────

    @staticmethod
    def lipinski_radar(df, ax):
        cats   = ["MW\n(≤500)", "LogP\n(≤5)", "HBD\n(≤5)", "HBA\n(≤10)",
                  "TPSA\n(≤140)", "RotBonds\n(≤10)"]
        cols   = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotatableBonds"]
        limits = [500, 5, 5, 10, 140, 10]
        N      = len(cats)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        for i, (_, row) in enumerate(df.iterrows()):
            vals = [min(row[c] / lim, 1.6) for c, lim in zip(cols, limits)]
            vals += vals[:1]
            color = plt.cm.tab10(i / max(len(df), 1))
            ax.plot(angles, vals, "o-", lw=2, color=color,
                    label=row.get("ID", f"C{i+1}"), alpha=0.8)
            ax.fill(angles, vals, alpha=0.15, color=color)

        # threshold ring
        ax.plot(angles, [1.0] * (N + 1), "--", lw=1.5,
                color="#E74C3C", label="Threshold")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, fontsize=9)
        ax.set_ylim(0, 1.7)
        ax.set_title("Lipinski / Veber Radar", fontweight="bold", pad=20)
        if len(df) <= 8:
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
        ax.grid(True, alpha=0.4)

    @staticmethod
    def mw_logp_scatter(df, ax):
        cmap = {0: PAL["green"], 1: PAL["orange"],
                2: PAL["red"],   3: PAL["purple"], 4: PAL["navy"]}
        colors = df["Lipinski_Violations"].map(cmap).fillna(PAL["grey"])
        sc = ax.scatter(df["MW"], df["LogP"], c=colors, s=90,
                        alpha=0.7, edgecolors="black", linewidths=0.8, zorder=3)
        ax.axvline(500, color=PAL["red"], ls="--", lw=2, label="MW = 500 Da")
        ax.axhline(5, color=PAL["blue"], ls="--", lw=2, label="LogP = 5")
        ax.fill_between([0, 500], [5, 5], [-10, -10],
                        alpha=0.07, color=PAL["green"], label="Drug-like space")
        ax.set_xlabel("Molecular Weight (Da)", fontweight="bold")
        ax.set_ylabel("LogP", fontweight="bold")
        ax.set_title("MW vs LogP — Lipinski Space", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Annotate outliers
        for _, r in df.iterrows():
            if r["Lipinski_Violations"] >= 2:
                ax.annotate(r.get("ID", ""), (r["MW"], r["LogP"]),
                            fontsize=7, ha="center", va="bottom",
                            xytext=(0, 5), textcoords="offset points")

    @staticmethod
    def tpsa_rotbonds(df, ax):
        colors = df["Veber_Pass"].map({True: PAL["green"], False: PAL["red"]})
        ax.scatter(df["RotatableBonds"], df["TPSA"], c=colors, s=90,
                   alpha=0.7, edgecolors="black", linewidths=0.8, zorder=3)
        ax.axvline(10,  color=PAL["red"],  ls="--", lw=2, label="RotBonds = 10")
        ax.axhline(140, color=PAL["blue"], ls="--", lw=2, label="TPSA = 140 Å²")
        ax.fill_between([0, 10], [0, 0], [140, 140],
                        alpha=0.07, color=PAL["green"], label="Veber space")
        ax.set_xlabel("Rotatable Bonds", fontweight="bold")
        ax.set_ylabel("TPSA (Å²)", fontweight="bold")
        ax.set_title("TPSA vs Rotatable Bonds — Veber Space", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Legend patches
        p1 = mpatches.Patch(color=PAL["green"], label="Veber PASS")
        p2 = mpatches.Patch(color=PAL["red"],   label="Veber FAIL")
        ax.legend(handles=[p1, p2], fontsize=8)

    @staticmethod
    def qed_histogram(df, ax):
        ax.hist(df["QED_Score"], bins=20, color=PAL["blue"],
                alpha=0.75, edgecolor="black", linewidth=1.2)
        ax.axvline(0.67, color=PAL["green"],  ls="--", lw=2, label="High (>0.67)")
        ax.axvline(0.34, color=PAL["orange"], ls="--", lw=2, label="Medium (>0.34)")
        ax.set_xlabel("QED Score", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("Drug-Likeness Distribution (QED)", fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

    @staticmethod
    def property_boxplot(df, ax):
        """Boxplot of key normalised properties side-by-side."""
        props  = ["MW", "LogP", "TPSA", "QED_Score", "RotatableBonds",
                  "HBD", "HBA", "Fsp3"]
        labels = ["MW/500", "LogP/5", "TPSA/140", "QED", "RotB/10",
                  "HBD/5", "HBA/10", "Fsp3"]
        limits = [500, 5, 140, 1, 10, 5, 10, 1]
        avail  = [(p, lbl, lim) for p, lbl, lim in zip(props, labels, limits)
                  if p in df.columns]
        data   = [df[p].dropna() / lim for p, _, lim in avail]
        lbls   = [lbl for _, lbl, _ in avail]

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        colors_box = [PAL["blue"], PAL["teal"], PAL["purple"],
                      PAL["green"], PAL["orange"], PAL["red"],
                      PAL["navy"], PAL["grey"]]
        for patch, c in zip(bp["boxes"], colors_box):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)

        ax.axhline(1.0, color=PAL["red"], ls="--", lw=1.5, label="Threshold = 1")
        ax.set_xticks(range(1, len(lbls) + 1))
        ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Normalised Value", fontweight="bold")
        ax.set_title("Property Distribution (Boxplot)", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    @staticmethod
    def comparison_heatmap(df, ax):
        num_cols = [c for c in [
            "MW", "LogP", "HBD", "HBA", "TPSA",
            "RotatableBonds", "QED_Score", "BBB_Score",
            "PlasmaProteinBinding_pct", "Fsp3", "MolRefractivity",
            "ESOL_LogS", "SyntheticAccessibility"
        ] if c in df.columns]

        if len(df) < 2 or not num_cols:
            ax.text(0.5, 0.5, "Need >1 compound for comparison",
                    ha="center", va="center", transform=ax.transAxes, fontsize=13)
            return

        dn = df[num_cols].copy().astype(float)
        for c in num_cols:
            rng = dn[c].max() - dn[c].min()
            if rng:
                dn[c] = (dn[c] - dn[c].min()) / rng

        ids = df["ID"].tolist() if "ID" in df.columns else [str(i) for i in range(len(df))]
        cmap = LinearSegmentedColormap.from_list(
            "rg", ["#E74C3C", "#F39C12", "#27AE60"])
        sns.heatmap(dn.T, cmap=cmap, ax=ax,
                    xticklabels=ids, yticklabels=num_cols,
                    linewidths=0.4, linecolor="#CCCCCC",
                    cbar_kws={"label": "Normalised Value", "shrink": 0.8})
        ax.set_xlabel("Compound", fontweight="bold")
        ax.set_title("ADMET Property Comparison Heatmap", fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    @staticmethod
    def parallel_coordinates(df, ax):
        if len(df) < 2:
            ax.text(0.5, 0.5, "Need >1 compound",
                    ha="center", va="center", transform=ax.transAxes)
            return
        cols = [c for c in ["MW", "LogP", "TPSA", "QED_Score", "HBD", "HBA"]
                if c in df.columns]
        dn   = df[cols].copy().astype(float)
        for c in cols:
            rng = dn[c].max() - dn[c].min()
            if rng:
                dn[c] = (dn[c] - dn[c].min()) / rng
        colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
        for (i, row), clr in zip(dn.iterrows(), colors):
            lbl = df.loc[i, "ID"] if "ID" in df.columns else f"C{i+1}"
            ax.plot(range(len(cols)), row.values, "o-",
                    lw=2, alpha=0.75, color=clr, label=lbl)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=35, ha="right")
        ax.set_ylabel("Normalised Value", fontweight="bold")
        ax.set_title("Parallel Coordinates — Compound Comparison", fontweight="bold")
        if len(df) <= 10:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def qed_bar(df, ax):
        ids    = df["ID"].tolist() if "ID" in df.columns else [str(i) for i in range(len(df))]
        scores = df["QED_Score"].tolist()
        colors_bar = [PAL["green"] if s > 0.67 else
                      PAL["orange"] if s > 0.34 else PAL["red"] for s in scores]
        bars = ax.bar(range(len(scores)), scores, color=colors_bar,
                      edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.axhline(0.67, color=PAL["green"],  ls="--", lw=2, label="High (>0.67)")
        ax.axhline(0.34, color=PAL["orange"], ls="--", lw=2, label="Medium (>0.34)")
        ax.set_xticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("QED Score", fontweight="bold")
        ax.set_title("QED Score per Compound", fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")


# ── GUI ──────────────────────────────────────────────────────────────────
class ADMETGUI:
    """
    Professional Tkinter GUI for ADMET Analyzer Pro.
    Supports single-SMILES and batch (CSV / SDF / TXT) analysis,
    2D molecular depictions, and tabbed publication-quality plots.
    """

    APP_TITLE   = "ADMET Analyzer Pro"
    APP_VERSION = "2.0"
    BG_MAIN     = "#F0F4F8"
    BG_PANEL    = "#FFFFFF"
    ACCENT      = "#2E86AB"
    TEXT_DARK   = "#2C3E50"

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{self.APP_TITLE} v{self.APP_VERSION}")
        self.root.geometry("1480x940")
        self.root.configure(bg=self.BG_MAIN)
        self.root.resizable(True, True)
        
        # Remove the default Tkinter icon (feather/leaf) by setting a blank icon
        try:
            # Create a transparent 1x1 pixel icon
            icon = tk.PhotoImage(width=1, height=1)
            self.root.iconphoto(True, icon)
        except Exception:
            pass  # If it fails, just continue without changing the icon

        if not RDKIT_AVAILABLE:
            messagebox.showerror(
                "RDKit Required",
                "RDKit is required.\n\nInstall with:\n  pip install rdkit pillow"
            )
            return

        self.df_results   = None
        self.smiles_list  = []
        self.compound_names = []

        ADMETPlotter.style()
        self._build_styles()
        self._build_ui()

    # ── Styles ────────────────────────────────────────────────────────────────

    def _build_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Title.TLabel",
                         font=("Arial", 17, "bold"),
                         foreground=self.ACCENT,
                         background=self.BG_MAIN)
        style.configure("Sub.TLabel",
                         font=("Arial", 9),
                         foreground="#7F8C8D",
                         background=self.BG_MAIN)
        style.configure("Section.TLabelframe",
                         background=self.BG_PANEL,
                         bordercolor="#C8D6E5",
                         relief="groove")
        style.configure("Section.TLabelframe.Label",
                         font=("Arial", 10, "bold"),
                         foreground=self.ACCENT,
                         background=self.BG_PANEL)
        style.configure("Accent.TButton",
                         font=("Arial", 10, "bold"),
                         padding=8)
        style.configure("Normal.TButton",
                         font=("Arial", 9),
                         padding=6)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=self.ACCENT, height=58)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        # No icon - only title text
        ttk.Label(hdr,
                  text=f"  {self.APP_TITLE}",
                  font=("Arial", 18, "bold"),
                  foreground="white", background=self.ACCENT
                  ).grid(row=0, column=0, sticky="w", padx=(15, 0), pady=8)
        
        ttk.Label(hdr,
                  text=f"35+ descriptors  |  Lipinski · Veber · Ghose · REOS · QED  |  v{self.APP_VERSION}",
                  font=("Arial", 9), foreground="#D6EAF8",
                  background=self.ACCENT
                  ).grid(row=0, column=1, sticky="e", padx=16)

        # ── Body PanedWindow ─────────────────────────────────────────────────
        body = tk.PanedWindow(self.root, orient=tk.HORIZONTAL,
                              bg=self.BG_MAIN, sashwidth=6,
                              sashrelief="flat", sashpad=2)
        body.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        # Left panel
        left = tk.Frame(body, bg=self.BG_MAIN, width=380)
        body.add(left, minsize=320)

        # Right panel
        right = tk.Frame(body, bg=self.BG_MAIN)
        body.add(right, minsize=700)

        self._build_left(left)
        self._build_right(right)

        # Status bar
        self.status_var = tk.StringVar(
            value="Ready — enter a SMILES or load a file (CSV / SDF / TXT)")
        sb = ttk.Label(self.root, textvariable=self.status_var,
                      relief=tk.SUNKEN, anchor=tk.W, padding=(6, 3))
        sb.grid(row=2, column=0, sticky="ew")

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left(self, parent):
        parent.columnconfigure(0, weight=1)

        # ── Input frame ───────────────────────────────────────────────────────
        inp = ttk.LabelFrame(parent, text="🔬  Input",
                              style="Section.TLabelframe", padding=10)
        inp.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 4))
        inp.columnconfigure(0, weight=1)

        ttk.Label(inp, text="SMILES:", font=("Arial", 9, "bold"),
                  background=self.BG_PANEL).grid(row=0, column=0, sticky="w")
        self.smiles_entry = ttk.Entry(inp, width=42, font=("Consolas", 9))
        self.smiles_entry.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self.smiles_entry.insert(0, "CC(=O)Oc1ccccc1C(=O)O")   # Aspirin

        ttk.Button(inp, text="⚗  Analyze Single",
                   command=self.analyze_single,
                   style="Accent.TButton"
                   ).grid(row=2, column=0, sticky="ew", pady=2)

        sep = ttk.Separator(inp, orient="horizontal")
        sep.grid(row=3, column=0, sticky="ew", pady=8)

        ttk.Button(inp, text="📂  Load File (CSV / SDF / TXT)",
                   command=self.load_file,
                   style="Normal.TButton"
                   ).grid(row=4, column=0, sticky="ew", pady=2)

        ttk.Button(inp, text="📊  Batch Analyze",
                   command=self.batch_analyze,
                   style="Normal.TButton"
                   ).grid(row=5, column=0, sticky="ew", pady=2)

        ttk.Button(inp, text="🖼  Show 2D Structures",
                   command=self.show_2d_structures,
                   style="Normal.TButton"
                   ).grid(row=6, column=0, sticky="ew", pady=2)

        ttk.Button(inp, text="💾  Export Results",
                   command=self.export_results,
                   style="Normal.TButton"
                   ).grid(row=7, column=0, sticky="ew", pady=(2, 4))

        # ── Quick summary ─────────────────────────────────────────────────────
        summ = ttk.LabelFrame(parent, text="📋  Quick Summary",
                               style="Section.TLabelframe", padding=10)
        summ.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        summ.columnconfigure(1, weight=1)

        self._summary_labels = {}
        rows_def = [
            ("MW",         "—  Da"),
            ("LogP",       "—"),
            ("TPSA",       "—  Å²"),
            ("QED",        "—"),
            ("Lipinski",   "—"),
            ("Veber",      "—"),
            ("Oral Bio",   "—"),
            ("BBB",        "—"),
            ("Tox Alerts", "—"),
        ]
        for i, (lbl, val) in enumerate(rows_def):
            ttk.Label(summ, text=lbl + ":",
                      font=("Arial", 9, "bold"),
                      background=self.BG_PANEL,
                      foreground=self.TEXT_DARK
                      ).grid(row=i, column=0, sticky="w", pady=1)
            sv = tk.StringVar(value=val)
            self._summary_labels[lbl] = sv
            ttk.Label(summ, textvariable=sv,
                      font=("Consolas", 9),
                      background=self.BG_PANEL
                      ).grid(row=i, column=1, sticky="w", padx=(8, 0), pady=1)

        # ── Info ──────────────────────────────────────────────────────────────
        info = ttk.LabelFrame(parent, text="ℹ️  References",
                               style="Section.TLabelframe", padding=8)
        info.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
        ref_text = (
            "Lipinski et al. Adv Drug Deliv Rev 2001\n"
            "Veber et al. J Med Chem 2002\n"
            "Ghose et al. J Comb Chem 1999\n"
            "Bickerton et al. Nat Chem 2012 (QED)\n"
            "Daina et al. Sci Rep 2017 (SwissADME)\n"
            "github.com/AbdelmoujoudFaris/ADMET_Analyzer"
        )
        ttk.Label(info, text=ref_text, font=("Arial", 8),
                  background=self.BG_PANEL,
                  foreground="#566573", justify="left",
                  wraplength=340
                  ).grid(row=0, column=0, sticky="w")

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # Results text area
        res_frame = ttk.LabelFrame(parent, text="📄  Analysis Report",
                                    style="Section.TLabelframe", padding=8)
        res_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 4))
        res_frame.columnconfigure(0, weight=1)
        res_frame.rowconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(
            res_frame, height=14, font=("Consolas", 9),
            bg="#FDFEFE", fg=self.TEXT_DARK,
            wrap=tk.WORD, relief="flat", borderwidth=1
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")

        # Notebook for plots
        viz_frame = ttk.LabelFrame(parent, text="📈  Visualisations",
                                    style="Section.TLabelframe", padding=6)
        viz_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_summary(self, p):
        def _yn(v): return "✓ Yes" if v else "✗ No"
        self._summary_labels["MW"].set(f"{p.get('MW', 0):.1f}  Da")
        self._summary_labels["LogP"].set(f"{p.get('LogP', 0):.2f}")
        self._summary_labels["TPSA"].set(f"{p.get('TPSA', 0):.1f}  Å²")
        self._summary_labels["QED"].set(f"{p.get('QED_Score', 0):.3f} "
                                         f"[{p.get('DrugLikeness', '—')}]")
        self._summary_labels["Lipinski"].set(
            f"{'✓ PASS' if p.get('Lipinski_Pass') else '✗ FAIL'} "
            f"({p.get('Lipinski_Violations', '?')} viol.)")
        self._summary_labels["Veber"].set(
            "✓ PASS" if p.get("Veber_Pass") else "✗ FAIL")
        self._summary_labels["Oral Bio"].set(
            "✓ Yes" if p.get("Oral_Bioavailability") else "✗ No")
        self._summary_labels["BBB"].set(p.get("BBB_Penetration", "—"))
        alerts = p.get("Toxicity_Alerts", ["None"])
        self._summary_labels["Tox Alerts"].set(
            "None" if alerts == ["None"] else ", ".join(alerts))

    def _clear_notebook(self):
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)

    def _add_fig_tab(self, fig, label):
        frame = ttk.Frame(self.notebook)
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        tb = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        tb.update()
        tb.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.notebook.add(frame, text=label)

    # ── Analysis ──────────────────────────────────────────────────────────────

    def analyze_single(self):
        smiles = self.smiles_entry.get().strip()
        if not smiles:
            messagebox.showwarning("Warning", "Please enter a SMILES string.")
            return
        try:
            self.status_var.set("Analysing…")
            self.root.update()
            p = ADMETCalculator.full_profile(smiles)
            self._update_summary(p)

            # Report
            self.results_text.delete(1.0, tk.END)
            self._write_single_report(smiles, p)

            self.df_results = pd.DataFrame([p])
            self.df_results["ID"]    = "Compound_1"
            self.df_results["SMILES"] = smiles
            self._make_single_plots()
            self.status_var.set("✓ Single analysis complete")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{e}")
            self.status_var.set("Error during analysis")

    def _write_single_report(self, smiles, p):
        W = self.results_text
        W.insert(tk.END, f"{'═'*80}\n")
        W.insert(tk.END, f" ADMET ANALYSIS REPORT   {datetime.now():%Y-%m-%d %H:%M}\n")
        W.insert(tk.END, f"{'═'*80}\n")
        W.insert(tk.END, f" SMILES : {smiles}\n")
        W.insert(tk.END, f" Formula: {p.get('MolFormula','—')}\n\n")

        def block(title): W.insert(tk.END, f"── {title} {'─'*(60-len(title))}\n")
        def row(k, v):    W.insert(tk.END, f"  {k:<35s}: {v}\n")

        block("PHYSICOCHEMICAL  (35+ descriptors)")
        row("Molecular Weight",               f"{p.get('MW',0):.2f}  Da")
        row("LogP (Crippen)",                 f"{p.get('LogP',0):.2f}")
        row("LogD estimate",                  f"{p.get('LogD_est',0):.2f}")
        row("TPSA",                           f"{p.get('TPSA',0):.2f}  Å²")
        row("H-Bond Donors",                   str(p.get('HBD','—')))
        row("H-Bond Acceptors",                str(p.get('HBA','—')))
        row("Rotatable Bonds",                 str(p.get('RotatableBonds','—')))
        row("Molar Refractivity",              f"{p.get('MolRefractivity',0):.2f}")
        row("Heavy Atoms",                     str(p.get('HeavyAtoms','—')))
        row("Total Atoms",                     str(p.get('NumAtoms','—')))
        row("Num Rings",                       str(p.get('NumRings','—')))
        row("Aromatic Rings",                  str(p.get('NumAromaticRings','—')))
        row("Aliphatic Rings",                 str(p.get('NumAliphaticRings','—')))
        row("Stereocentres",                   str(p.get('StereoCenters','—')))
        row("Fsp3",                            f"{p.get('Fsp3',0):.3f}")
        row("Formal Charge",                    str(p.get('FormalCharge','—')))
        row("ESOL LogS (solubility)",          f"{p.get('ESOL_LogS',0):.2f}")
        row("Water Solubility class",          str(p.get('WaterSolubility','—')))

        W.insert(tk.END, "\n")
        block("DRUG-LIKENESS RULES")
        row("Lipinski Ro5 violations",  str(p.get('Lipinski_Violations','—')))
        row("Lipinski Rule",            "✓ PASS" if p.get('Lipinski_Pass') else "✗ FAIL")
        row("Veber Rule",               "✓ PASS" if p.get('Veber_Pass')    else "✗ FAIL")
        row("Ghose Filter",             "✓ PASS" if p.get('Ghose_Pass')    else "✗ FAIL")
        row("REOS Filter",              "✓ PASS" if p.get('REOS_Pass')     else "✗ FAIL")
        row("Lead-Like Filter",         "✓ PASS" if p.get('LeadLike_Pass') else "✗ FAIL")
        row("Egan Egg",                 "✓ PASS" if p.get('Egan_Pass')     else "✗ FAIL")
        row("QED Score",                f"{p.get('QED_Score',0):.4f}  [{p.get('DrugLikeness','—')}]")
        row("Synthetic Accessibility",  f"{p.get('SyntheticAccessibility',0):.1f}  (1=easy, 10=hard)")

        W.insert(tk.END, "\n")
        block("ABSORPTION")
        row("HIA Probability",          f"{p.get('HIA_Probability','—')}%")
        row("Caco-2 Permeability",      str(p.get('Caco2_Permeability','—')))
        row("Pgp Substrate",            str(p.get('Pgp_Substrate','—')))
        row("Oral Bioavailability",     "✓ Yes" if p.get('Oral_Bioavailability') else "✗ No")
        row("Bioavailability F20",      "✓ Yes" if p.get('BioAvailability_F20') else "✗ No")
        row("Bioavailability F30",      "✓ Yes" if p.get('BioAvailability_F30') else "✗ No")

        W.insert(tk.END, "\n")
        block("DISTRIBUTION")
        row("BBB Penetration",          str(p.get('BBB_Penetration','—')))
        row("BBB Score",                f"{p.get('BBB_Score',0):.3f}")
        row("Plasma Protein Binding",   f"{p.get('PlasmaProteinBinding_pct',0):.1f}%")
        row("Volume of Distribution",   str(p.get('VolumeDistribution','—')))
        row("CNS MPO",                  str(p.get('CNS_MPO','—')))

        W.insert(tk.END, "\n")
        block("METABOLISM")
        row("CYP3A4 Substrate",         "Yes" if p.get('CYP3A4_Substrate') else "No")
        row("CYP2D6 Substrate",         "Yes" if p.get('CYP2D6_Substrate') else "No")
        row("CYP2C9 Substrate",         "Yes" if p.get('CYP2C9_Substrate') else "No")
        row("CYP3A4 Inhibitor",         "Yes" if p.get('CYP3A4_Inhibitor') else "No")
        row("CYP2D6 Inhibitor",         "Yes" if p.get('CYP2D6_Inhibitor') else "No")
        row("Metabolic Stability",      str(p.get('MetabolicStability','—')))

        W.insert(tk.END, "\n")
        block("EXCRETION")
        row("Renal Clearance",          str(p.get('RenalClearance','—')))
        row("Half-life Category",       str(p.get('HalfLife_Category','—')))
        row("Biliary Clearance",        str(p.get('BiliaryClearance','—')))

        W.insert(tk.END, "\n")
        block("TOXICITY")
        row("Structural Alerts",        ", ".join(p.get('Toxicity_Alerts', ['None'])))
        row("PAINS Alert",              "Yes" if p.get('PAINS_Alert') else "No")
        row("PAINS Descriptions",       ", ".join(p.get('PAINS_Descriptions', ['None'])))
        row("Ames Mutagenicity",        "Yes" if p.get('Ames_Mutagenicity') else "No")
        row("Hepatotoxicity Risk",      str(p.get('Hepatotoxicity_Risk','—')))
        row("hERG Risk",                str(p.get('hERG_Risk','—')))
        row("Cramer Toxicity Class",    str(p.get('Cramer_Class','—')))
        W.insert(tk.END, f"\n{'═'*80}\n")

    # ── Batch ─────────────────────────────────────────────────────────────────

    def load_file(self):
        fn = filedialog.askopenfilename(
            title="Select compound file",
            filetypes=[("All supported", "*.csv;*.sdf;*.txt"),
                       ("CSV files", "*.csv"),
                       ("SDF files", "*.sdf"),
                       ("Text files", "*.txt"),
                       ("All files",  "*.*")]
        )
        if not fn:
            return
        try:
            self.smiles_list, self.compound_names = [], []
            p = Path(fn)
            if p.suffix.lower() == ".csv":
                df = pd.read_csv(fn)
                sc = next((c for c in df.columns
                           if c.upper() in ("SMILES", "SMILE", "SMI")), None)
                if sc is None:
                    messagebox.showerror("Error", "CSV must have a 'SMILES' column.")
                    return
                self.smiles_list = df[sc].tolist()
                nc = next((c for c in df.columns
                           if c.upper() in ("NAME", "ID", "COMPOUND", "MOLECULE")), None)
                self.compound_names = df[nc].tolist() if nc else \
                    [f"Compound_{i+1}" for i in range(len(self.smiles_list))]

            elif p.suffix.lower() == ".sdf":
                suppl = Chem.SDMolSupplier(fn)
                for i, mol in enumerate(suppl):
                    if mol:
                        self.smiles_list.append(Chem.MolToSmiles(mol))
                        nm = mol.GetProp("_Name") \
                            if mol.HasProp("_Name") else f"Compound_{i+1}"
                        self.compound_names.append(nm)

            elif p.suffix.lower() == ".txt":
                with open(fn) as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = [x.strip() for x in line.split(",")]
                        if len(parts) == 2:
                            m0 = Chem.MolFromSmiles(parts[0])
                            m1 = Chem.MolFromSmiles(parts[1])
                            if m0:
                                self.smiles_list.append(parts[0])
                                self.compound_names.append(
                                    parts[1] if not m1 else f"Compound_{i+1}")
                            elif m1:
                                self.smiles_list.append(parts[1])
                                self.compound_names.append(parts[0])
                        else:
                            self.smiles_list.append(parts[0])
                            self.compound_names.append(f"Compound_{i+1}")

            # Deduplicate
            seen, us, un = set(), [], []
            for s, n in zip(self.smiles_list, self.compound_names):
                if s not in seen:
                    seen.add(s); us.append(s); un.append(n)
            self.smiles_list = us
            self.compound_names = un

            self.status_var.set(
                f"✓ Loaded {len(self.smiles_list)} unique compounds — ready for batch analysis")
            messagebox.showinfo("File Loaded",
                                f"Loaded {len(self.smiles_list)} unique compounds\n"
                                f"from {p.name}\n\nClick 'Batch Analyze' to proceed.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def batch_analyze(self):
        if not self.smiles_list:
            messagebox.showwarning("Warning", "Load a file first.")
            return
        try:
            self.status_var.set("Batch analysis in progress…")
            self.root.update()
            results, failed = [], []
            total = len(self.smiles_list)

            for i, (smi, name) in enumerate(zip(self.smiles_list, self.compound_names)):
                try:
                    p = ADMETCalculator.full_profile(smi)
                    p["SMILES"] = smi
                    p["ID"]     = name
                    p["Index"]  = i + 1
                    results.append(p)
                except Exception as e:
                    failed.append((name, smi, str(e)))
                if (i + 1) % 10 == 0:
                    self.status_var.set(f"Analysed {i+1}/{total}…")
                    self.root.update()

            self.df_results = pd.DataFrame(results)
            self.results_text.delete(1.0, tk.END)
            self._write_batch_report(results, failed)
            self._make_batch_plots()
            self.status_var.set(
                f"✓ Batch complete: {len(results)} compounds  |  {len(failed)} failed")
        except Exception as e:
            messagebox.showerror("Error", f"Batch analysis failed:\n{e}")

    def _write_batch_report(self, results, failed):
        W  = self.df_results
        rt = self.results_text

        rt.insert(tk.END, f"{'╔'+'═'*78+'╗'}\n")
        rt.insert(tk.END, f"║{'BATCH ADMET ANALYSIS REPORT':^78}║\n")
        rt.insert(tk.END, f"║{datetime.now():%Y-%m-%d %H:%M':^78}║\n")
        rt.insert(tk.END, f"{'╚'+'═'*78+'╝'}\n\n")

        n = len(results)
        rt.insert(tk.END, f"📊 OVERVIEW\n")
        rt.insert(tk.END, f"   Total compounds analysed : {n}\n")
        if failed:
            rt.insert(tk.END, f"   Failed                  : {len(failed)}\n")
        rt.insert(tk.END, "\n")

        rt.insert(tk.END, "✓ FILTER COMPLIANCE\n")
        for col, label in [
            ("Lipinski_Pass",       "Lipinski Ro5  (≤1 viol.)"),
            ("Veber_Pass",          "Veber Rule"),
            ("Ghose_Pass",          "Ghose Filter"),
            ("REOS_Pass",           "REOS Filter"),
            ("LeadLike_Pass",       "Lead-like Filter"),
            ("Egan_Pass",           "Egan Egg"),
            ("Oral_Bioavailability","Oral Bioavailability"),
        ]:
            if col in W.columns:
                cnt = W[col].sum()
                rt.insert(tk.END,
                    f"   {label:<35}: {cnt:3d}/{n} ({cnt/n*100:5.1f}%)\n")

        rt.insert(tk.END, "\n📈 PROPERTY STATISTICS  (Mean ± SD)\n")
        for col, lbl, unit in [
            ("MW",                   "Molecular Weight",          "Da"),
            ("LogP",                 "LogP",                      ""),
            ("TPSA",                 "TPSA",                      "Å²"),
            ("HBD",                  "H-Bond Donors",             ""),
            ("HBA",                  "H-Bond Acceptors",          ""),
            ("RotatableBonds",       "Rotatable Bonds",           ""),
            ("QED_Score",            "QED Score",                 ""),
            ("SyntheticAccessibility","Synthetic Accessibility",  ""),
            ("ESOL_LogS",            "ESOL LogS",                 ""),
            ("Fsp3",                 "Fsp3",                      ""),
        ]:
            if col in W.columns:
                rt.insert(tk.END,
                    f"   {lbl:<35}: {W[col].mean():7.3f} ± {W[col].std():6.3f}"
                    f"{'  '+unit if unit else ''}\n")

        rt.insert(tk.END, "\n🏆 TOP 10 COMPOUNDS (QED Score)\n")
        top = W.nlargest(min(10, n), "QED_Score")
        for _, row in top.iterrows():
            rt.insert(tk.END,
                f"   {int(row['Index']):2d}. {row['ID']:<30}  "
                f"QED={row['QED_Score']:.3f}  MW={row['MW']:.1f}  "
                f"LogP={row['LogP']:.2f}  "
                f"{'✓Lip' if row['Lipinski_Pass'] else '✗Lip'}  "
                f"{'✓Veb' if row['Veber_Pass'] else '✗Veb'}\n")

        rt.insert(tk.END, f"\n{'─'*80}\n")
        rt.insert(tk.END, "INDIVIDUAL COMPOUND DETAILS\n")
        rt.insert(tk.END, f"{'─'*80}\n\n")
        for _, row in W.iterrows():
            smi_short = row['SMILES'][:65] + ("…" if len(row['SMILES']) > 65 else "")
            rt.insert(tk.END,
                f"[{int(row['Index'])}] {row['ID']}\n"
                f"    SMILES  : {smi_short}\n"
                f"    MW={row['MW']:.1f} Da  LogP={row['LogP']:.2f}  "
                f"TPSA={row['TPSA']:.1f} Å²  QED={row['QED_Score']:.3f}  "
                f"Fsp3={row.get('Fsp3',0):.2f}\n"
                f"    Rules   : Lip[{'✓' if row['Lipinski_Pass'] else '✗'}]  "
                f"Veb[{'✓' if row['Veber_Pass'] else '✗'}]  "
                f"Gho[{'✓' if row['Ghose_Pass'] else '✗'}]  "
                f"REOS[{'✓' if row.get('REOS_Pass',False) else '✗'}]  "
                f"Oral[{'✓' if row['Oral_Bioavailability'] else '✗'}]\n"
                f"    BBB={row['BBB_Penetration']}  PPB={row.get('PlasmaProteinBinding_pct',0):.0f}%  "
                f"CYP3A4={'Sub' if row.get('CYP3A4_Substrate') else 'N/A'}  "
                f"hERG={row.get('hERG_Risk','—')}  "
                f"Tox={', '.join(row.get('Toxicity_Alerts',['None']))}\n\n"
            )

        if failed:
            rt.insert(tk.END, "⚠ FAILED COMPOUNDS\n")
            for name, smi, err in failed:
                rt.insert(tk.END, f"   • {name}: {err}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────

    def _make_single_plots(self):
        self._clear_notebook()
        df = self.df_results

        # Overview
        fig1 = Figure(figsize=(13, 4.5))
        gs   = fig1.add_gridspec(1, 3, wspace=0.35)

        ax_r = fig1.add_subplot(gs[0], projection="polar")
        ADMETPlotter.lipinski_radar(df, ax_r)

        ax_b = fig1.add_subplot(gs[1])
        props  = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotatableBonds"]
        vals   = [df[p].iloc[0] for p in props if p in df.columns]
        labels = [p for p in props if p in df.columns]
        cols   = [PAL["blue"], PAL["teal"], PAL["green"],
                  PAL["orange"], PAL["purple"], PAL["red"]]
        ax_b.barh(labels, vals, color=cols[:len(vals)],
                  edgecolor="black", linewidth=0.8, alpha=0.8)
        ax_b.set_xlabel("Value", fontweight="bold")
        ax_b.set_title("Key Properties", fontweight="bold")
        ax_b.grid(axis="x", alpha=0.3)

        ax_c = fig1.add_subplot(gs[2])
        criteria = ["Lipinski", "Veber", "Ghose", "REOS", "Oral Bio"]
        pcols    = ["Lipinski_Pass", "Veber_Pass", "Ghose_Pass",
                    "REOS_Pass", "Oral_Bioavailability"]
        pass_v   = [bool(df[c].iloc[0]) if c in df.columns else False
                    for c in pcols]
        clrs     = [PASS_COLOR if p else FAIL_COLOR for p in pass_v]
        ax_c.barh(criteria, [1] * len(criteria), color=clrs,
                  edgecolor="black", linewidth=0.8)
        ax_c.set_xlim(0, 1)
        ax_c.set_xlabel("Pass / Fail", fontweight="bold")
        ax_c.set_title("Filter Results", fontweight="bold")
        for i, (c, pv) in enumerate(zip(criteria, pass_v)):
            ax_c.text(0.5, i, "✓" if pv else "✗",
                      ha="center", va="center",
                      fontsize=14, color="white", fontweight="bold")
        self._add_fig_tab(fig1, "Overview")

        # Boxplot
        fig2 = Figure(figsize=(13, 5))
        ax2  = fig2.add_subplot(111)
        ADMETPlotter.property_boxplot(df, ax2)
        self._add_fig_tab(fig2, "Properties")

    def _make_batch_plots(self):
        self._clear_notebook()
        df = self.df_results

        # 1 – Drug-like space
        fig1 = Figure(figsize=(13, 5))
        gs1  = fig1.add_gridspec(1, 2, wspace=0.35)
        ADMETPlotter.mw_logp_scatter(df, fig1.add_subplot(gs1[0]))
        ADMETPlotter.tpsa_rotbonds(df, fig1.add_subplot(gs1[1]))
        self._add_fig_tab(fig1, "Drug-like Space")

        # 2 – Drug-likeness
        fig2 = Figure(figsize=(13, 5))
        gs2  = fig2.add_gridspec(1, 2, wspace=0.35)
        ADMETPlotter.qed_histogram(df, fig2.add_subplot(gs2[0]))
        ADMETPlotter.qed_bar(df, fig2.add_subplot(gs2[1]))
        self._add_fig_tab(fig2, "Drug-Likeness (QED)")

        # 3 – Boxplot (NEW)
        fig3 = Figure(figsize=(13, 5))
        ADMETPlotter.property_boxplot(df, fig3.add_subplot(111))
        self._add_fig_tab(fig3, "Property Boxplot")

        # 4 – Heatmap
        fig4 = Figure(figsize=(13, 6))
        ADMETPlotter.comparison_heatmap(df, fig4.add_subplot(111))
        self._add_fig_tab(fig4, "Comparison Heatmap")

        # 5 – Parallel Coordinates
        fig5 = Figure(figsize=(13, 5.5))
        ADMETPlotter.parallel_coordinates(df, fig5.add_subplot(111))
        self._add_fig_tab(fig5, "Parallel Coordinates")

        # 6 – Radar
        fig6 = Figure(figsize=(8, 8))
        ADMETPlotter.lipinski_radar(df, fig6.add_subplot(111, projection="polar"))
        self._add_fig_tab(fig6, "Lipinski Radar")

        # 7 – Filter compliance bar
        fig7 = Figure(figsize=(10, 5))
        ax7  = fig7.add_subplot(111)
        filters = [("Lipinski_Pass", "Lipinski Ro5"),
                   ("Veber_Pass",    "Veber"),
                   ("Ghose_Pass",    "Ghose"),
                   ("REOS_Pass",     "REOS"),
                   ("LeadLike_Pass", "Lead-like"),
                   ("Egan_Pass",     "Egan Egg"),
                   ("Oral_Bioavailability", "Oral Bio")]
        lbls  = [l for c, l in filters if c in df.columns]
        pass_ = [df[c].sum() for c, l in filters if c in df.columns]
        fail_ = [len(df) - v for v in pass_]
        x     = np.arange(len(lbls))
        ax7.bar(x, pass_, color=PASS_COLOR, alpha=0.8,
                label="Pass", edgecolor="black", linewidth=0.7)
        ax7.bar(x, fail_, bottom=pass_, color=FAIL_COLOR, alpha=0.8,
                label="Fail", edgecolor="black", linewidth=0.7)
        ax7.set_xticks(x)
        ax7.set_xticklabels(lbls, rotation=30, ha="right")
        ax7.set_ylabel("Number of Compounds", fontweight="bold")
        ax7.set_title("Filter Compliance Summary", fontweight="bold")
        ax7.legend(fontsize=9)
        ax7.grid(axis="y", alpha=0.3)
        for xi, (p, tot) in enumerate(zip(pass_, [len(df)]*len(pass_))):
            ax7.text(xi, tot + 0.2, f"{p/tot*100:.0f}%",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
        self._add_fig_tab(fig7, "Filter Compliance")

    # ── 2D Structures ─────────────────────────────────────────────────────────

    def show_2d_structures(self):
        if not PIL_AVAILABLE:
            messagebox.showwarning("PIL Required",
                "Install Pillow for 2D depictions:\n  pip install pillow")
            return
        if self.df_results is None:
            messagebox.showwarning("No Data", "Run analysis first.")
            return

        mols   = [Chem.MolFromSmiles(s) for s in self.df_results["SMILES"]
                  if Chem.MolFromSmiles(s) is not None]
        labels = list(self.df_results["ID"])

        if not mols:
            messagebox.showerror("Error", "No valid molecules found.")
            return

        win = tk.Toplevel(self.root)
        win.title("2D Molecular Structures")
        win.geometry("1200x700")
        win.configure(bg=self.BG_MAIN)

        ttk.Label(win, text="2D Molecular Depictions",
                  font=("Arial", 14, "bold"),
                  background=self.BG_MAIN,
                  foreground=self.ACCENT).pack(pady=(10, 4))

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Scrollable canvas
        canvas = tk.Canvas(frame, bg="white")
        vsb    = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        hsb    = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        try:
            mpr    = min(4, len(mols))
            grid_img = MolDepiction.grid_to_pil(
                mols, labels=labels[:len(mols)],
                mols_per_row=mpr, sub_img_size=(260, 200)
            )
            from PIL import ImageTk
            tk_img = ImageTk.PhotoImage(grid_img)
            canvas.create_image(0, 0, anchor="nw", image=tk_img)
            canvas.config(scrollregion=(0, 0, grid_img.width, grid_img.height))
            canvas._img_ref = tk_img   # prevent GC
        except Exception as e:
            canvas.create_text(300, 200, text=f"Depiction error:\n{e}",
                               font=("Arial", 11), fill="red")

        ttk.Button(win, text="💾  Save as PNG",
                   command=lambda: self._save_2d_image(grid_img),
                   style="Normal.TButton").pack(pady=6)

    def _save_2d_image(self, img):
        fn = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if fn:
            img.save(fn, dpi=(300, 300))
            messagebox.showinfo("Saved", f"Saved to:\n{fn}")

    # ── Export ────────────────────────────────────────────────────────────────

    def export_results(self):
        if self.df_results is None:
            messagebox.showwarning("Warning", "No results to export.")
            return
        fn = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not fn:
            return
        try:
            # Flatten list columns
            df_exp = self.df_results.copy()
            for col in df_exp.columns:
                if df_exp[col].dtype == object:
                    df_exp[col] = df_exp[col].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else x)
            df_exp.to_csv(fn, index=False)

            base = Path(fn)
            # Text report
            rep_fn = base.with_name(base.stem + "_report.txt")
            with open(rep_fn, "w", encoding="utf-8") as f:
                f.write(self.results_text.get(1.0, tk.END))

            # Figures
            saved_figs = 0
            for tab_id in self.notebook.tabs():
                lbl    = self.notebook.tab(tab_id, "text").replace(" ", "_")
                fig_fn = base.with_name(f"{base.stem}_{lbl}.png")
                try:
                    frame = self.notebook.nametowidget(tab_id)
                    for w in frame.winfo_children():
                        if isinstance(w, tk.Widget) and hasattr(w, "master"):
                            pass
                    # retrieve figure from canvas
                    for child in frame.winfo_children():
                        if hasattr(child, "figure"):
                            child.figure.savefig(
                                fig_fn, dpi=300, bbox_inches="tight")
                            saved_figs += 1
                            break
                except Exception:
                    pass

            messagebox.showinfo("Export Complete",
                f"✓ Data      : {base.name}\n"
                f"✓ Report    : {rep_fn.name}\n"
                f"✓ Figures   : {saved_figs} PNG files")
            self.status_var.set(f"✓ Exported → {base.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app  = ADMETGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
