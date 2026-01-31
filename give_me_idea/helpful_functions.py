import json
import os
from typing import Union

import pandas as pd
from pandas.io.formats.style import Styler

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")
brain_url = os.environ.get("BRAIN_URL", "https://platform.worldquantbrain.com")


def make_clickable_alpha_id(alpha_id: str) -> str:
    """
    Create a clickable HTML link for an alpha ID.

    Args:
        alpha_id (str): The ID of the alpha.

    Returns:
        str: An HTML string containing a clickable link to the alpha's page on the platform.
    """

    url = brain_url + "/alpha/"
    return f'<a href="{url}{alpha_id}">{alpha_id}</a>'


def prettify_result(
    result: list, detailed_tests_view: bool = False, clickable_alpha_id: bool = False
) -> Union[pd.DataFrame, Styler]:
    """
    Combine and format simulation results into a single DataFrame for analysis.

    Args:
        result (list): A list of dictionaries containing simulation results.
        detailed_tests_view (bool, optional): If True, include detailed test results. Defaults to False.
        clickable_alpha_id (bool, optional): If True, make alpha IDs clickable. Defaults to False.

    Returns:
        pandas.DataFrame or pandas.io.formats.style.Styler: A DataFrame containing formatted results,
        optionally with clickable alpha IDs.
    """
    list_of_is_stats = [result[x]["is_stats"] for x in range(len(result)) if result[x]["is_stats"] is not None]
    is_stats_df = pd.concat(list_of_is_stats).reset_index(drop=True)
    is_stats_df = is_stats_df.sort_values("fitness", ascending=False)

    expressions = {
        result[x]["alpha_id"]: (
            {
                "selection": result[x]["simulate_data"]["selection"],
                "combo": result[x]["simulate_data"]["combo"],
            }
            if result[x]["simulate_data"]["type"] == "SUPER"
            else result[x]["simulate_data"]["regular"]
        )
        for x in range(len(result))
        if result[x]["is_stats"] is not None
    }
    expression_df = pd.DataFrame(list(expressions.items()), columns=["alpha_id", "expression"])

    list_of_is_tests = [result[x]["is_tests"] for x in range(len(result)) if result[x]["is_tests"] is not None]
    is_tests_df = pd.concat(list_of_is_tests, sort=True).reset_index(drop=True)
    is_tests_df = is_tests_df[is_tests_df["result"] != "WARNING"]
    if detailed_tests_view:
        cols = ["limit", "result", "value"]
        is_tests_df["details"] = is_tests_df[cols].to_dict(orient="records")
        is_tests_df = is_tests_df.pivot(index="alpha_id", columns="name", values="details").reset_index()
    else:
        is_tests_df = is_tests_df.pivot(index="alpha_id", columns="name", values="result").reset_index()

    alpha_stats = pd.merge(is_stats_df, expression_df, on="alpha_id")
    alpha_stats = pd.merge(alpha_stats, is_tests_df, on="alpha_id")
    alpha_stats = alpha_stats.drop(columns=alpha_stats.columns[(alpha_stats == "PENDING").any()])
    alpha_stats.columns = alpha_stats.columns.str.replace("(?<=[a-z])(?=[A-Z])", "_", regex=True).str.lower()
    if clickable_alpha_id:
        return alpha_stats.style.format({"alpha_id": lambda x: make_clickable_alpha_id(str(x))})
    return alpha_stats


def concat_pnl(result: list) -> pd.DataFrame:
    """
    Combine PnL results from multiple alphas into a single DataFrame.

    Args:
        result (list): A list of dictionaries containing simulation results with PnL data.

    Returns:
        pandas.DataFrame: A DataFrame containing combined PnL data for all alphas.
    """
    list_of_pnls = [result[x]["pnl"] for x in range(len(result)) if result[x]["pnl"] is not None]
    pnls_df = pd.concat(list_of_pnls).reset_index()

    return pnls_df


def concat_is_tests(result: list) -> pd.DataFrame:
    """
    Combine in-sample test results from multiple alphas into a single DataFrame.

    Args:
        result (list): A list of dictionaries containing simulation results with in-sample test data.

    Returns:
        pandas.DataFrame: A DataFrame containing combined in-sample test results for all alphas.
    """
    is_tests_list = [result[x]["is_tests"] for x in range(len(result)) if result[x]["is_tests"] is not None]
    is_tests_df = pd.concat(is_tests_list, sort=True).reset_index(drop=True)
    return is_tests_df


def save_simulation_result(result: dict) -> None:
    """
    Save the simulation result to a JSON file in the 'simulation_results' folder.

    Args:
        result (dict): A dictionary containing the simulation result for an alpha.
    """

    alpha_id = result["id"]
    region = result["settings"]["region"]
    folder_path = "simulation_results/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}")

    os.makedirs(folder_path, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(result, file)


def save_pnl(pnl_df: pd.DataFrame, alpha_id: str, region: str) -> None:
    """
    Save the PnL data for an alpha to a CSV file in the 'alphas_pnl' folder.

    Args:
        pnl_df (pandas.DataFrame): The DataFrame containing PnL data.
        alpha_id (str): The ID of the alpha.
        region (str): The region for which the PnL data was generated.
    """

    folder_path = "alphas_pnl/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}.csv")
    os.makedirs(folder_path, exist_ok=True)

    pnl_df.to_csv(file_path)


def save_yearly_stats(yearly_stats: pd.DataFrame, alpha_id: str, region: str):
    """
    Save the yearly statistics for an alpha to a CSV file in the 'yearly_stats' folder.

    Args:
        yearly_stats (pandas.DataFrame): The DataFrame containing yearly statistics.
        alpha_id (str): The ID of the alpha.
        region (str): The region for which the statistics were generated.
    """

    folder_path = "yearly_stats/"
    file_path = os.path.join(folder_path, f"{alpha_id}_{region}.csv")
    os.makedirs(folder_path, exist_ok=True)

    yearly_stats.to_csv(file_path, index=False)


def expand_dict_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Expand dictionary columns in a DataFrame into separate columns.

    Args:
        data (pandas.DataFrame): The input DataFrame with dictionary columns.

    Returns:
        pandas.DataFrame: A new DataFrame with expanded columns.
    """
    dict_columns = list(filter(lambda x: isinstance(data[x].iloc[0], dict), data.columns))
    new_columns = pd.concat(
        [data[col].apply(pd.Series).rename(columns=lambda x: f"{col}_{x}") for col in dict_columns],
        axis=1,
    )

    data = pd.concat([data, new_columns], axis=1)
    return data
