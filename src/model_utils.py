# src/model_utils.py
import pandas as pd

def _normalize_str(x):
    return x.strip().lower() if isinstance(x, str) else x

def _parse_pests(pests_cell):
    if pd.isna(pests_cell):
        return set()
    return set(p.strip() for p in str(pests_cell).split(',') if p.strip())

def recommend_next_crops_with_score(current_crop, original_df, top_n=5):
    cur_rows = original_df[original_df['Crop Name'] == current_crop]
    if cur_rows.empty:
        print(f"⚠️ Crop '{current_crop}' not found.")
        return []
    cur = cur_rows.iloc[0]

    current_season = cur['Season']
    current_soil = cur['Soil Type']
    current_pests = _parse_pests(cur['Pests'])

    next_season_map = {"Kharif": "Rabi", "Rabi": "Zaid", "Zaid": "Kharif"}
    next_season = next_season_map.get(current_season, "Unknown")

    results = []
    for c in original_df['Crop Name'].unique():
        if c == current_crop: continue
        row = original_df[original_df['Crop Name'] == c].iloc[0]

        crop_season = row['Season']
        crop_soil = row['Soil Type']
        crop_pests = _parse_pests(row['Pests'])

        num_common = len(current_pests.intersection(crop_pests))
        pest_score = 50.0 if len(current_pests) == 0 else ((len(current_pests) - num_common) / len(current_pests)) * 50.0
        soil_score = 25.0 if _normalize_str(crop_soil) == _normalize_str(current_soil) else 0.0
        season_score = 25.0 if _normalize_str(crop_season) == _normalize_str(next_season) else 0.0

        results.append({
            "Crop": c,
            "Season": crop_season,
            "Soil Type": crop_soil,
            "Pests in Common": num_common,
            "Common Pests": ", ".join(sorted(list(current_pests.intersection(crop_pests)))) if num_common else "None",
            "Plantability Score": round(pest_score + soil_score + season_score, 2)
        })

    results = sorted(results, key=lambda r: (r["Plantability Score"], -r["Pests in Common"]), reverse=True)
    return results[:top_n]