import getpass
import json
import os
import sys
from typing import List

import pandas as pd

# Ensure we can import ace_lib from the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import ace_lib  # noqa: E402


def prompt_credentials() -> tuple[str, str]:
    """Prompt user for platform credentials."""
    email = input("Enter BRAIN Email: ").strip()
    while not email:
        email = input("Email is required. Enter BRAIN Email: ").strip()

    password = getpass.getpass("Enter BRAIN Password: ").strip()
    while not password:
        password = getpass.getpass("Password is required. Enter BRAIN Password: ").strip()

    return email, password


def fetch_all_combinations(session: ace_lib.SingleSession) -> pd.DataFrame:
    """Return all valid instrument/region/delay/universe combos from platform settings."""
    options_df = ace_lib.get_instrument_type_region_delay(session)
    if options_df is None or options_df.empty:
        raise RuntimeError("No simulation options fetched; cannot enumerate datasets.")
    return options_df


def fetch_datasets_for_combo(
    session: ace_lib.SingleSession,
    instrument_type: str,
    region: str,
    delay: int,
    universe: str,
) -> pd.DataFrame:
    """Fetch datasets for one combination (theme ALL to include both theme true/false)."""
    df = ace_lib.get_datasets(
        session,
        instrument_type=instrument_type,
        region=region,
        delay=delay,
        universe=universe,
        theme="ALL",
    )
    if df is None:
        return pd.DataFrame()

    df = df.copy()
    df["param_instrument_type"] = instrument_type
    df["param_region"] = region
    df["param_delay"] = delay
    df["param_universe"] = universe
    df["combo_key"] = df.apply(
        lambda row: f"{instrument_type}-{region}-D{delay}-{universe}",
        axis=1,
    )
    return df


def merge_and_deduplicate(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge fetched datasets and deduplicate by dataset id, keeping all combo metadata."""
    combined = pd.concat([df for df in datasets if not df.empty], ignore_index=True)
    if combined.empty:
        return combined

    # Aggregate availability combos per dataset id
    availability = (
        combined.groupby("id")["combo_key"]
        .agg(lambda x: " | ".join(sorted(set(x))))
        .rename("available_in")
        .reset_index()
    )

    # Drop duplicate rows by dataset id, keep first occurrence of other columns
    unique_df = combined.drop_duplicates(subset=["id"]).copy()
    unique_df = unique_df.merge(availability, on="id", how="left")

    # Sort for readability
    sort_cols = [col for col in ["category", "subcategory", "id"] if col in unique_df.columns]
    if sort_cols:
        # Ensure sort keys are hashable/strings to avoid unhashable dict errors
        for col in sort_cols:
            unique_df[col] = unique_df[col].apply(
                lambda v: v
                if pd.isna(v) or isinstance(v, (int, float, str, bool))
                else json.dumps(v, ensure_ascii=False, sort_keys=True)
            )
        unique_df = unique_df.sort_values(sort_cols).reset_index(drop=True)

    return unique_df


def main():
    print("=== Fetch All BRAIN Datasets (all regions/universes/delays) ===")

    email, password = prompt_credentials()

    # Monkey-patch ace_lib credential retrieval so start_session uses provided credentials
    ace_lib.get_credentials = lambda: (email, password)

    print("Logging in...")
    try:
        session = ace_lib.start_session()
        print("Login successful.")
    except Exception as exc:
        print(f"Login failed: {exc}")
        return

    print("Fetching valid instrument/region/delay/universe combinations from platform settings...")
    try:
        options_df = fetch_all_combinations(session)
    except Exception as exc:
        print(f"Failed to fetch simulation options: {exc}")
        return

    all_datasets: List[pd.DataFrame] = []
    total_combos = 0

    for _, row in options_df.iterrows():
        instrument_type = row.get("InstrumentType")
        region = row.get("Region")
        delay = row.get("Delay")
        universes = row.get("Universe") or []

        for universe in universes:
            total_combos += 1
            print(f"[{total_combos}] Fetching datasets for {instrument_type} / {region} / D{delay} / {universe}...")
            try:
                df = fetch_datasets_for_combo(session, instrument_type, region, delay, universe)
                print(f"  -> Retrieved {len(df)} rows")
                all_datasets.append(df)
            except Exception as exc:
                print(f"  -> Failed for {instrument_type}-{region}-D{delay}-{universe}: {exc}")

    result_df = merge_and_deduplicate(all_datasets)

    if result_df.empty:
        print("No datasets fetched; nothing to save.")
        return

    output_path = os.path.join(SCRIPT_DIR, "all_datasets_full.csv")
    result_df.to_csv(output_path, index=False)
    print(f"Saved {len(result_df)} unique datasets to {output_path}")


if __name__ == "__main__":
    main()