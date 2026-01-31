import getpass
import os
import sys
from typing import List

import pandas as pd

# Make ace_lib importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import ace_lib  # noqa: E402


def prompt_credentials() -> tuple[str, str]:
    email = input("Enter BRAIN Email: ").strip()
    while not email:
        email = input("Email is required. Enter BRAIN Email: ").strip()

    password = getpass.getpass("Enter BRAIN Password: ").strip()
    while not password:
        password = getpass.getpass("Password is required. Enter BRAIN Password: ").strip()

    return email, password


def fetch_operators(session: ace_lib.SingleSession) -> pd.DataFrame:
    df = ace_lib.get_operators(session)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Choose an identifier column robustly
    id_col = "id" if "id" in df.columns else None
    if id_col is None:
        if "name" in df.columns:
            id_col = "name"
        else:
            id_col = "_row_id"
            df[id_col] = df.index

    # Re-aggregate scopes so each operator id is unique
    if "scope" in df.columns:
        scope_map = (
            df.groupby(id_col)["scope"]
            .agg(lambda x: sorted(set([item for item in x if pd.notna(item)])))
            .rename("scopes")
            .reset_index()
        )
    else:
        scope_map = pd.DataFrame({id_col: df[id_col].unique(), "scopes": [[] for _ in range(df[id_col].nunique())]})

    unique_df = df.drop(columns=["scope"], errors="ignore").drop_duplicates(subset=[id_col]).merge(
        scope_map, on=id_col, how="left"
    )

    # Sort for readability
    sort_cols: List[str] = [col for col in ["category", "subcategory", "name", id_col] if col in unique_df.columns]
    if sort_cols:
        unique_df = unique_df.sort_values(sort_cols).reset_index(drop=True)

    return unique_df


def main():
    print("=== Fetch All BRAIN Operators ===")

    email, password = prompt_credentials()
    ace_lib.get_credentials = lambda: (email, password)

    print("Logging in...")
    try:
        session = ace_lib.start_session()
        print("Login successful.")
    except Exception as exc:
        print(f"Login failed: {exc}")
        return

    print("Fetching operators...")
    try:
        operators_df = fetch_operators(session)
    except Exception as exc:
        print(f"Failed to fetch operators: {exc}")
        return

    if operators_df.empty:
        print("No operators returned; nothing to save.")
        return

    output_path = os.path.join(SCRIPT_DIR, "all_operators.csv")
    operators_df.to_csv(output_path, index=False)
    print(f"Saved {len(operators_df)} operators to {output_path}")


if __name__ == "__main__":
    main()