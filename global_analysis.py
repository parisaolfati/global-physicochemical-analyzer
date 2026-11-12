# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Amino acid 3-letter to 1-letter mapping
AA_MAPPING = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# Physicochemical grouping
PHYSICO_GROUPS = {
    'Aromatic': ['W', 'H', 'Y', 'F'],
    'Positively Charged': ['K', 'R'],
    'Negatively Charged': ['D', 'E'],
    'Aliphatic/Hydrophobic': ['V', 'I', 'L', 'M'],
    'Small/Polar': ['S', 'T', 'C', 'N', 'Q', 'A', 'G', 'P']
}
AA_TO_GROUP = {aa: group for group, aas in PHYSICO_GROUPS.items() for aa in aas}


def fix_numeric(val):
    """Convert input string with slashes to numeric float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.replace('/', '.').strip()
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def assign_region(rel_pos):
    """Assign structural region based on relative sequence position."""
    if rel_pos <= 0.33:
        return 'N-Terminal'
    elif rel_pos <= 0.66:
        return 'Core/Middle'
    else:
        return 'C-Terminal'


def load_all_data(root_path: Path, target_file: str) -> pd.DataFrame:
    """Load and aggregate data from all project subdirectories."""
    print(f"Loading and compiling dataset from: {root_path.resolve()}")
    all_data = []

    for folder_path in root_path.iterdir():
        if not folder_path.is_dir():
            continue
        file_path = folder_path / target_file
        if not file_path.exists():
            continue
        try:
            df = pd.read_excel(file_path)
            # Ensure expected columns
            if len(df.columns) >= 7:
                df.columns = ['Mutation', 'col_1_complex', 'col_1_peptide',
                              'DeltaG_binding', 'DeltaDeltaG_binding',
                              'DeltaG_mutation', 'Position'][:len(df.columns)]
            else:
                continue

            df['DeltaDeltaG_binding'] = df['DeltaDeltaG_binding'].apply(fix_numeric)
            df['Position'] = df['Position'].apply(fix_numeric)

            # Exclude wild-type reference and null values
            df_mut = df[~df['Mutation'].astype(str).str.contains('WTref', case=False, na=False)].copy()
            df_mut = df_mut.dropna(subset=['Position', 'DeltaDeltaG_binding'])
            if df_mut.empty:
                continue

            # Extract introduced amino acid and group
            df_mut['Introduced_AA_3L'] = df_mut['Mutation'].apply(lambda x: str(x).split('_')[0].upper())
            df_mut['Introduced_AA'] = df_mut['Introduced_AA_3L'].map(AA_MAPPING)
            df_mut['Physicochemical_Group'] = df_mut['Introduced_AA'].map(AA_TO_GROUP)

            # Relative sequence position and structural region
            max_len = df_mut['Position'].max()
            if max_len > 0:
                df_mut['Relative_Position'] = df_mut['Position'] / max_len
            else:
                df_mut['Relative_Position'] = 0.0
            df_mut['Structural_Region'] = df_mut['Relative_Position'].apply(assign_region)

            df_mut['Peptide'] = folder_path.name
            all_data.append(df_mut)
        except Exception as e:
            print(f"Error processing {folder_path.name}: {e}")

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def run_global_analysis(root_dir: Path, target_file: str, output_excel: str):
    """Run full aggregation, statistical analysis, and generate summary plots and reports."""
    full_df = load_all_data(root_dir, target_file)
    if full_df.empty:
        print("No valid data found to analyze.")
        return

    print(f"Successfully compiled {len(full_df)} mutation records across projects.")

    # 1. Amino Acid Ranking Summary
    aa_summary = full_df.groupby('Introduced_AA').agg(
        Mean_DDG=('DeltaDeltaG_binding', 'mean'),
        SD_DDG=('DeltaDeltaG_binding', 'std'),
        SEM_DDG=('DeltaDeltaG_binding', lambda x: x.std() / np.sqrt(x.count()) if len(x) > 1 else 0),
        Beneficial_Count=('DeltaDeltaG_binding', lambda x: (x < 0).sum()),
        Total_Tested=('DeltaDeltaG_binding', 'count')
    ).reset_index()

    aa_summary['Success_Rate (%)'] = round(
        (aa_summary['Beneficial_Count'] / aa_summary['Total_Tested']) * 100, 2)
    aa_summary = aa_summary.sort_values(by='Mean_DDG')

    plt.figure(figsize=(11, 5))
    colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in aa_summary['Mean_DDG']]
    plt.bar(aa_summary['Introduced_AA'], aa_summary['Mean_DDG'],
            yerr=aa_summary['SEM_DDG'], capsize=4, color=colors, edgecolor='black', linewidth=0.5,
            error_kw={'ecolor': '#333333', 'elinewidth': 1.2, 'capthick': 1.2})
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel('Mean DeltaDeltaG (kcal/mol)')
    plt.xlabel('Introduced Amino Acid')
    plt.title('Global Amino Acid Mutation Impact')
    plt.tight_layout()
    plt.savefig(root_dir / "GLOBAL_1_amino_acid_impact.png", dpi=300)
    plt.close()

    # 2. Structural Region Summary
    def calc_region_stats(ddg):
        n = len(ddg)
        if n == 0:
            return pd.Series({'Destructive_Percent': 0, 'Destructive_SEM': 0, 'Beneficial_Percent': 0, 'Beneficial_SEM': 0})
        dest_p = (ddg > 0).sum() / n
        ben_p = (ddg < 0).sum() / n
        dest_sem = np.sqrt(dest_p * (1 - dest_p) / n) * 100
        ben_sem = np.sqrt(ben_p * (1 - ben_p) / n) * 100
        return pd.Series({
            'Destructive_Percent': dest_p * 100,
            'Destructive_SEM': dest_sem,
            'Beneficial_Percent': ben_p * 100,
            'Beneficial_SEM': ben_sem
        })

    region_summary = full_df.groupby('Structural_Region')['DeltaDeltaG_binding']\
        .apply(calc_region_stats).unstack()\
        .reindex(['N-Terminal', 'Core/Middle', 'C-Terminal']).reset_index()

    plt.figure(figsize=(8, 5))
    x_indices = np.arange(len(region_summary))
    width = 0.35
    plt.bar(x_indices - width / 2, region_summary['Destructive_Percent'], width=width,
            yerr=region_summary['Destructive_SEM'], capsize=4, color='#e74c3c', label='Destructive (ΔΔG > 0)')
    plt.bar(x_indices + width / 2, region_summary['Beneficial_Percent'], width=width,
            yerr=region_summary['Beneficial_SEM'], capsize=4, color='#2ecc71', label='Beneficial (ΔΔG < 0)')
    plt.xticks(x_indices, region_summary['Structural_Region'])
    plt.ylabel('Percentage (%)')
    plt.title('Mutation Outcome by Structural Region')
    plt.legend()
    plt.tight_layout()
    plt.savefig(root_dir / "GLOBAL_2_structural_regions.png", dpi=300)
    plt.close()

    # Save to Master Excel Report
    master_output = root_dir / output_excel
    with pd.ExcelWriter(master_output, engine="openpyxl") as writer:
        aa_summary.to_excel(writer, sheet_name="Amino_Acid_Ranking", index=False)
        region_summary.to_excel(writer, sheet_name="Structural_Regions", index=False)
        full_df.to_excel(writer, sheet_name="All_Compiled_Mutations", index=False)

    print(f"Global analysis completed successfully!")
    print(f"Report saved to: {master_output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run global physicochemical and structural analysis across peptide screening results.")
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        type=Path,
        help="Root directory containing project subfolders (default: current directory)."
    )
    parser.add_argument(
        "-t", "--target",
        default="all_positions_summary.xlsx",
        help="Target summary filename in each subfolder (default: all_positions_summary.xlsx)."
    )
    parser.add_argument(
        "-o", "--output",
        default="Global_Physicochemical_Analysis_Report.xlsx",
        help="Output master Excel filename (default: Global_Physicochemical_Analysis_Report.xlsx)."
    )
    args = parser.parse_args()
    
    run_global_analysis(args.root_dir, args.target, args.output)
