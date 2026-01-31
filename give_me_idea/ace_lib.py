import getpass
import json
import logging
import os
import threading
import time
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import urljoin

import pandas as pd
import requests
import tqdm
from helpful_functions import (
    expand_dict_columns,
    save_pnl,
    save_simulation_result,
    save_yearly_stats,
)

DEV = False


class SingleSession(requests.Session):
    _instance = None
    _lock = threading.Lock()
    _relogin_lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        if not self._initialized:
            super(SingleSession, self).__init__(*args, **kwargs)
            self._initialized = True

    def get_relogin_lock(self):
        return self._relogin_lock


def setup_logger() -> logging.Logger:
    """
    This function sets up a logger that writes log messages to the console and,
    if the global variable DEV is set to True, also to a file named 'ace.log'.

    Returns:
        logger (logging.Logger): The configured logger object.

    The logger's name is set to 'ace.log'. The level of the logger and the console handler
    is set to INFO if DEV is True, and WARNING otherwise. The format for the log messages
    is: 'asctime' - 'name' - 'levelname' - 'message'.
    """
    logger = logging.getLogger("ace")
    level = logging.DEBUG if DEV else logging.INFO

    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("ace.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


DEFAULT_CONFIG = {
    "get_pnl": False,
    "get_stats": False,
    "save_pnl_file": False,
    "save_stats_file": False,
    "save_result_file": False,
    "check_submission": False,
    "check_self_corr": False,
    "check_prod_corr": False,
}

brain_api_url = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")


def get_credentials() -> tuple[str, str]:
    """
    Retrieve or prompt for platform credentials.

    This function attempts to read credentials from a JSON file in the user's home directory.
    If the file doesn't exist or is empty, it prompts the user to enter credentials and saves them.

    Returns:
        tuple: A tuple containing the email and password.

    Raises:
        json.JSONDecodeError: If the credentials file exists but contains invalid JSON.
    """

    return ("xxx@xxxx.com", "xxxxxx")


def start_session() -> SingleSession:
    """
    Start a new session with the WorldQuant BRAIN platform.

    This function authenticates the user, handles biometric authentication if required,
    and creates a new session.

    Returns:
        SingleSession: An authenticated session object.

    Raises:
        requests.exceptions.RequestException: If there's an error during the authentication process.
    """

    s = SingleSession()
    s.auth = get_credentials()
    r = s.post(brain_api_url + "/authentication")
    logger.debug(f"New session created (ID: {id(s)}) with authentication response: {r.status_code}, {r.json()}")
    if r.status_code == requests.status_codes.codes.unauthorized:
        if r.headers["WWW-Authenticate"] == "persona":
            print(
                "Complete biometrics authentication and press any key to continue: \n"
                + urljoin(r.url, r.headers["Location"])
                + "\n"
            )
            input()
            s.post(urljoin(r.url, r.headers["Location"]))

            while True:
                if s.post(urljoin(r.url, r.headers["Location"])).status_code != 201:
                    input(
                        "Biometrics authentication is not complete. Please try again and press any key when completed \n"
                    )
                else:
                    break
        else:
            logger.error("\nIncorrect email or password\n")
            with open(
                os.path.join(os.path.expanduser("~"), "secrets/platform-brain.json"),
                "w",
            ) as file:
                json.dump({}, file)
            return start_session()
    return s


def check_session_timeout(s: SingleSession) -> int:
    """
    Check if the current session has timed out.

    Args:
        s (SingleSession): The current session object.

    Returns:
        int: The number of seconds until the session expires, or 0 if the session has expired or an error occurred.
    """

    authentication_url = brain_api_url + "/authentication"
    try:
        result = s.get(authentication_url).json()["token"]["expiry"]
        logger.debug(f"Session (ID: {id(s)}) timeout check result: {result}")
        return result
    except Exception:
        return 0


def generate_alpha(
    regular: Optional[str] = None,
    selection: Optional[str] = None,
    combo: Optional[str] = None,
    alpha_type: Literal["REGULAR", "SUPER"] = "REGULAR",
    region: str = "USA",
    universe: str = "TOP3000",
    delay: Literal[0, 1] = 1,
    decay: int = 0,
    neutralization: str = "INDUSTRY",
    truncation: float = 0.08,
    pasteurization: Literal["ON", "OFF"] = "ON",
    test_period: str = "P0Y0M0D",
    unit_handling: Literal["VERIFY"] = "VERIFY",
    nan_handling: Literal["ON", "OFF"] = "OFF",
    max_trade: Literal["ON", "OFF"] = "OFF",
    selection_handling: str = "POSITIVE",
    selection_limit: int = 100,
    visualization: bool = False,
) -> dict:
    """
    Generate an alpha dictionary for simulation. If alpha_type='REGULAR',
    function generates alpha dictionary using regular input. If alpha_type='SUPER',
    function generates alpha dictionary using selection and combo inputs.

    Args:
        regular (str, optional): The regular alpha expression.
        selection (str, optional): The selection expression for super alphas.
        combo (str, optional): The combo expression for super alphas.
        alpha_type (str, optional): The type of alpha ("REGULAR" or "SUPER"). Defaults to "REGULAR".
        region (str, optional): The region for the alpha. Defaults to "USA".
        universe (str, optional): The universe for the alpha. Defaults to "TOP3000".
        delay (int, optional): The delay for the alpha. Defaults to 1.
        decay (int, optional): The decay for the alpha. Defaults to 0.
        neutralization (str, optional): The neutralization method. Defaults to "INDUSTRY".
        truncation (float, optional): The truncation value. Defaults to 0.08.
        pasteurization (str, optional): The pasteurization setting. Defaults to "ON".
        test_period (str, optional): The test period. Defaults to "P0Y0M0D".
        unit_handling (str, optional): The unit handling method. Defaults to "VERIFY".
        nan_handling (str, optional): The NaN handling method. Defaults to "OFF".
        max_trade (str, optional): The max trade method. Defaults to "OFF".
        selection_handling (str, optional): The selection handling method for super alphas. Defaults to "POSITIVE".
        selection_limit (int, optional): The selection limit for super alphas. Defaults to 100.
        visualization (bool, optional): Whether to include visualization. Defaults to False.

    Returns:
        dict: A dictionary containing the alpha configuration for simulation.

    Raises:
        ValueError: If an invalid alpha_type is provided.
    """

    settings = {
        "instrumentType": "EQUITY",
        "region": region,
        "universe": universe,
        "delay": delay,
        "decay": decay,
        "neutralization": neutralization,
        "truncation": truncation,
        "pasteurization": pasteurization,
        "testPeriod": test_period,
        "unitHandling": unit_handling,
        "nanHandling": nan_handling,
        "maxTrade": max_trade,
        "language": "FASTEXPR",
        "visualization": visualization,
    }
    if alpha_type == "REGULAR":
        simulation_data = {
            "type": alpha_type,
            "settings": settings,
            "regular": regular,
        }
    elif alpha_type == "SUPER":
        simulation_data = {
            "type": alpha_type,
            "settings": {
                **settings,
                "selectionHandling": selection_handling,
                "selectionLimit": selection_limit,
            },
            "combo": combo,
            "selection": selection,
        }
    else:
        logger.error("alpha_type should be REGULAR or SUPER")
        return {}
    return simulation_data


def check_session_and_relogin(s: SingleSession) -> SingleSession:
    """
    Checks for session timeout and if less than 2000 seconds are remaining,
    it attempts to start a new session.

    Parameters:
        s (SingleSession): The current session object.

    Returns:
        s (SingleSession): The original session object if it hasn't timed out,
        otherwise a new session object.

    If the remaining session time is less than 2000 seconds, the function
    attempts to start a new session using the `start_session()` function.
    If `start_session()` fails on the first attempt, it waits for 100 seconds
    and then tries again. The function then returns the new session object.
    """
    with s.get_relogin_lock():
        if check_session_timeout(s) < 2000:
            logger.debug('Session less than 2000 seconds')
            try:
                s = start_session()
            except Exception:
                logger.info('Trying re-login, wait 100 seconds')
                time.sleep(100)
                s = start_session()
        logger.debug(f"Session (ID: {id(s)}) after check and relogin")
    return s


def start_simulation(s: SingleSession, simulate_data: Union[list[dict], dict]) -> requests.Response:
    """
    Start a simulation with the provided simulation data.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_data (dict): A dictionary containing the simulation parameters.

    Returns:
        requests.Response: The response object from the simulation start request.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """
    simulate_response = s.post(brain_api_url + "/simulations", json=simulate_data)
    return simulate_response


def simulation_progress(
    s: SingleSession,
    simulate_response: requests.Response,
) -> dict:
    """
    Monitor the progress of a simulation and return the result when complete.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_response (requests.Response): The response from starting the simulation.

    Returns:
        dict: A dictionary containing the completion status and simulation result.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """
    if simulate_response.status_code // 100 != 2:
        logger.warning(f'Simulation failed. {simulate_response.text}, Status code: {simulate_response.status_code}')
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    retry_count = 0
    while True:
        simulation_progress_response = s.get(simulation_progress_url)
        if simulation_progress_response.status_code // 100 != 2:
            logger.error(
                f'Simulation {simulation_progress_url}, Status code: {simulation_progress_response.status_code}, Retry'
            )
            time.sleep(30)
            retry_count += 1
            if retry_count <= 2:
                continue
            else:
                logger.error(
                    f'Simulation {simulation_progress_url} failed, Status code: {simulation_progress_response.status_code}'
                )
                error_flag = True
                break
        if simulation_progress_response.headers.get("Retry-After", 0) == 0:
            if simulation_progress_response.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break

        time.sleep(float(simulation_progress_response.headers["Retry-After"]))

    if error_flag:
        logger.error(f"Simulation failed. {simulation_progress_response.json()}")
        return {"completed": False, "result": {}}

    alpha = simulation_progress_response.json().get("alpha", 0)
    if alpha == 0:
        logger.warning(
            f'Simulation {simulation_progress_response.json().get("id")} failed. {simulation_progress_response.json()}'
        )
        return {"completed": False, "result": {}}
    simulation_result = get_simulation_result_json(s, alpha)
    if len(simulation_result) == 0:
        return {"completed": False, "result": {}}
    return {"completed": True, "result": simulation_result}


def get_simulation_result_json(s: SingleSession, alpha_id: str) -> dict:
    """
    Retrieve the full simulation result for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        dict: A dictionary containing the full simulation result.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """
    if alpha_id is None:
        return {}
    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id)
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    try:
        return result.json()
    except Exception:
        logger.error(f"alpha_id {alpha_id}, {result.headers}, {result.text}, {result.status_code}")
        return {}
    return s.get(brain_api_url + "/alphas/" + alpha_id).json()


def multisimulation_progress(
    s: SingleSession,
    simulate_response: requests.Response,
) -> dict:
    """
    Monitor the progress of multiple simulations and return the results when complete.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_response (requests.Response): The response from starting the simulations.

    Returns:
        dict: A dictionary containing the completion status and simulation results.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """
    if simulate_response.status_code // 100 != 2:
        logger.warning(f'Simulation failed. {simulate_response.text}, Status code: {simulate_response.status_code}')
        return {"completed": False, "result": {}}

    simulation_progress_url = simulate_response.headers["Location"]
    error_flag = False
    while True:
        simulation_progress_response = s.get(simulation_progress_url)
        if simulation_progress_response.status_code // 100 != 2:
            time.sleep(30)
        if simulation_progress_response.headers.get("Retry-After", 0) == 0:
            if simulation_progress_response.json().get("status", "ERROR") == "ERROR":
                error_flag = True
            break

        time.sleep(float(simulation_progress_response.headers["Retry-After"]))

    children = simulation_progress_response.json().get("children", 0)

    if error_flag:
        if children == 0:
            logger.error(f"Simulation failed. {simulation_progress_response.json()}")
            return {"completed": False, "result": {}}
        for child in children:
            child_progress = s.get(brain_api_url + "/simulations/" + child)
            logger.error(f"Child Simulation failed: {child_progress.json()}")
        return {"completed": False, "result": {}}

    if len(children) == 0:
        logger.warning(
            f'Multi-Simulation {simulation_progress_response.json().get("id")} failed. {simulation_progress_response.json()}'
        )
        return {"completed": False, "result": {}}
    children_list = []
    for child in children:
        child_progress = s.get(brain_api_url + "/simulations/" + child)
        alpha = child_progress.json().get("alpha", 0)
        if alpha == 0:
            logger.warning(f'Child-Simulation {child_progress.json().get("id")} failed. {child_progress.json()}')
            return {"completed": False, "result": {}}
        child_result = get_simulation_result_json(s, alpha)
        children_list.append(child_result)
    return {"completed": True, "result": children_list}


def get_prod_corr(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the production correlation data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the production correlation data.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/correlations/prod")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        logger.warning(f"Failed to get production correlation for alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()
    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    prod_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    prod_corr_df["alpha_max_prod_corr"] = result.json()["max"]
    prod_corr_df["alpha_min_prod_corr"] = result.json()["min"]

    return prod_corr_df


def check_prod_corr_test(s: SingleSession, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
    """
    Check if the alpha's production correlation passes a specified threshold.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.
        threshold (float, optional): The correlation threshold. Defaults to 0.7.

    Returns:
        pandas.DataFrame: A DataFrame containing the test result.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    prod_corr_df = get_prod_corr(s, alpha_id)
    if prod_corr_df.empty:
        result = [
            {
                "test": "PROD_CORRELATION",
                "result": "NONE",
                "limit": threshold,
                "value": None,
                "alpha_id": alpha_id,
            }
        ]
    else:
        value = prod_corr_df[prod_corr_df.alphas > 0]["max"].max()
        result = [
            {
                "test": "PROD_CORRELATION",
                "result": "PASS" if value <= threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id,
            }
        ]
    return pd.DataFrame(result)


def get_self_corr(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the self-correlation data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the self-correlation data.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/correlations/self")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("records", 0) == 0:
        logger.warning(f"Failed to get self correlation for alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()

    records_len = len(result.json()["records"])
    if records_len == 0:
        logger.warning(f"No self correlation for alpha_id {alpha_id}")
        return pd.DataFrame()

    columns = [dct["name"] for dct in result.json()["schema"]["properties"]]
    self_corr_df = pd.DataFrame(result.json()["records"], columns=columns).assign(alpha_id=alpha_id)
    self_corr_df["alpha_max_self_corr"] = result.json()["max"]
    self_corr_df["alpha_min_self_corr"] = result.json()["min"]

    return self_corr_df


def check_self_corr_test(s: SingleSession, alpha_id: str, threshold: float = 0.7) -> pd.DataFrame:
    """
    Check if the alpha's self-correlation passes a specified threshold.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.
        threshold (float, optional): The correlation threshold. Defaults to 0.7.

    Returns:
        pandas.DataFrame: A DataFrame containing the test result.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    self_corr_df = get_self_corr(s, alpha_id)
    if self_corr_df.empty:
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS",
                "limit": threshold,
                "value": 0,
                "alpha_id": alpha_id,
            }
        ]
    else:
        value = self_corr_df["correlation"].max()
        result = [
            {
                "test": "SELF_CORRELATION",
                "result": "PASS" if value < threshold else "FAIL",
                "limit": threshold,
                "value": value,
                "alpha_id": alpha_id,
            }
        ]
    return pd.DataFrame(result)


def get_check_submission(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the submission check results for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the submission check results.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """

    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/check")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("is", 0) == 0:
        logger.warning(f"Cant check submission alpha_id {alpha_id}. {result.json()}")
        return pd.DataFrame()

    checks_df = pd.DataFrame(result.json()["is"]["checks"]).assign(alpha_id=alpha_id)

    return checks_df


def simulate_multi_alpha(
    s: SingleSession,
    simulate_data_list: list,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """
    Simulate a list of alphas using multi-simulation.

    This function checks the session timeout, starts a new session if necessary,
    initiates the simulation, monitors its progress, and sets alpha properties
    upon completion.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_data (dict): A list of dictionaries, each containing the simulation parameters for the alpha.
            These should include all necessary information such as alpha type, settings, and expressions.

    Returns:
        list: A list of dictionaries, each containing:
            - 'alpha_id' (str): The ID of the simulated alpha if successful, None otherwise.
            - 'simulate_data' (dict): The original simulation data provided.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """

    s = check_session_and_relogin(s)
    if len(simulate_data_list) == 1:
        return [simulate_single_alpha(s, simulate_data_list[0])]
    simulate_response = start_simulation(s, simulate_data_list)
    simulation_result = multisimulation_progress(s, simulate_response)

    if not simulation_result["completed"]:
        return [{"alpha_id": None, "simulate_data": x} for x in simulate_data_list]
    result = [
        {
            "alpha_id": x["id"],
            "simulate_data": {
                "type": x["type"],
                "settings": x["settings"],
                "regular": x["regular"]["code"],
            },
        }
        for x in simulation_result["result"]
    ]
    if tags:
        _ = [set_alpha_properties(s, x["id"], tags=tags) for x in simulation_result["result"]]
    else:
        _ = [set_alpha_properties(s, x["id"]) for x in simulation_result["result"]]
    return result


def get_specified_alpha_stats(
    s: SingleSession,
    alpha_id: Union[str, None],
    simulate_data: dict,
    get_pnl: bool = False,
    get_stats: bool = False,
    save_pnl_file: bool = False,
    save_stats_file: bool = False,
    save_result_file: bool = False,
    check_submission: bool = False,
    check_self_corr: bool = False,
    check_prod_corr: bool = False,
) -> dict:
    """
    Retrieve and process specified statistics for a given alpha.

    Args:
        s (SingleSession): The authenticated session object.
        alpha_id (str): The ID of the alpha to retrieve statistics for.
        simulate_data (dict): The original simulation data for the alpha.
        get_pnl (bool, optional): Whether to retrieve PnL data. Defaults to False.
        get_stats (bool, optional): Whether to retrieve yearly stats. Defaults to False.
        save_pnl_file (bool, optional): Whether to save PnL data to a file. Defaults to False.
        save_stats_file (bool, optional): Whether to save yearly stats to a file. Defaults to False.
        save_result_file (bool, optional): Whether to save the simulation result to a file. Defaults to False.
        check_submission (bool, optional): Whether to perform submission checks. Defaults to False.
        check_self_corr (bool, optional): Whether to check self-correlation. Defaults to False.
        check_prod_corr (bool, optional): Whether to check production correlation. Defaults to False.

    Returns:
        dict: A dictionary containing various statistics and information about the alpha.

    Raises:
        requests.exceptions.RequestException: If there's an error retrieving data from the API.
    """
    pnl = None
    stats = None
    s = check_session_and_relogin(s)
    logger.debug(f"Session (ID: {id(s)}) used in get_specified_alpha_stats for alpha_id: {alpha_id}")
    if alpha_id is None:
        return {
            "alpha_id": None,
            "simulate_data": simulate_data,
            "is_stats": None,
            "pnl": pnl,
            "stats": stats,
            "is_tests": None,
            "train": None,
            "test": None,
        }

    result = get_simulation_result_json(s, alpha_id)
    try:
        region = result["settings"]["region"]
        is_stats = pd.DataFrame([{key: value for key, value in result['is'].items() if key != 'checks'}]).assign(
            alpha_id=alpha_id
        )
    except Exception as e:
        logger.error(f"Failed to retrieve simulation result for alpha_id {alpha_id}: {result}, {e}")
    train = result["train"]
    test = result["test"]
    is_stats = pd.DataFrame([{key: value for key, value in result["is"].items() if key != "checks"}]).assign(
        alpha_id=alpha_id
    )

    if get_pnl:
        pnl = get_alpha_pnl(s, alpha_id)
        if save_pnl_file:
            save_pnl(pnl, alpha_id, region)

    if get_stats:
        stats = get_alpha_yearly_stats(s, alpha_id)
        if save_stats_file:
            save_yearly_stats(stats, alpha_id, region)

    if save_result_file:
        save_simulation_result(result)

    is_tests = pd.DataFrame(result["is"]["checks"]).assign(alpha_id=alpha_id)

    if check_submission:
        is_tests = get_check_submission(s, alpha_id)

        return {
            "alpha_id": alpha_id,
            "simulate_data": simulate_data,
            "is_stats": is_stats,
            "pnl": pnl,
            "stats": stats,
            "is_tests": is_tests,
            "train": train,
            "test": test,
        }

    if check_self_corr and not check_submission:
        self_corr_test = check_self_corr_test(s, alpha_id)
        is_tests = (
            pd.concat([is_tests, pd.DataFrame([self_corr_test])], ignore_index=True)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )
    if check_prod_corr and not check_submission:
        prod_corr_test = check_prod_corr_test(s, alpha_id)
        is_tests = (
            pd.concat([is_tests, pd.DataFrame([prod_corr_test])], ignore_index=True)
            .drop_duplicates(subset=["test"], keep="last")
            .reset_index(drop=True)
        )

    return {
        "alpha_id": alpha_id,
        "simulate_data": simulate_data,
        "is_stats": is_stats,
        "pnl": pnl,
        "stats": stats,
        "is_tests": is_tests,
        "train": train,
        "test": test,
    }


def simulate_single_alpha(
    s: SingleSession,
    simulate_data: dict,
) -> dict:
    """
    Simulate a single alpha using the provided session and simulation data.

    This function checks the session timeout, starts a new session if necessary,
    initiates the simulation, monitors its progress, and sets alpha properties
    upon completion.

    Args:
        s (SingleSession): An authenticated session object.
        simulate_data (dict): A dictionary containing the simulation parameters for the alpha.
            This should include all necessary information such as alpha type, settings, and expressions.

    Returns:
        dict: A dictionary containing:
            - 'alpha_id' (str): The ID of the simulated alpha if successful, None otherwise.
            - 'simulate_data' (dict): The original simulation data provided.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """

    s = check_session_and_relogin(s)
    simulate_response = start_simulation(s, simulate_data)
    simulation_result = simulation_progress(s, simulate_response)

    if not simulation_result["completed"]:
        return {"alpha_id": None, "simulate_data": simulate_data}
    set_alpha_properties(s, simulation_result["result"]["id"])
    return {
        "alpha_id": simulation_result["result"]["id"],
        "simulate_data": simulate_data,
    }


def simulate_alpha_list(
    s: SingleSession,
    alpha_list: list,
    limit_of_concurrent_simulations: int = 3,
    simulation_config: dict = DEFAULT_CONFIG,
) -> list:
    """
    Simulate a list of alphas concurrently.

    Args:
        s (SingleSession): The authenticated session object.
        alpha_list (list): A list of alpha configurations to simulate.
        limit_of_concurrent_simulations (int, optional): The maximum number of concurrent simulations. Defaults to 3.
        simulation_config (dict, optional): Configuration for the simulation. Defaults to DEFAULT_CONFIG.

    Returns:
        list: A list of dictionaries containing simulation results for each alpha.

    Raises:
        requests.exceptions.RequestException: If there's an error during the simulation process.
    """
    if (limit_of_concurrent_simulations < 1) or (limit_of_concurrent_simulations > 8):
        logger.warning("Limit of concurrent simulation should be 1..8, will be set to 3")
        limit_of_concurrent_simulations = 3

    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        with tqdm.tqdm(total=len(alpha_list)) as pbar:
            for result in pool.imap_unordered(partial(simulate_single_alpha, s), alpha_list):
                result_list.append(result)
                pbar.update()

    stats_list_result = []

    def func(x):
        return get_specified_alpha_stats(s, x["alpha_id"], x["simulate_data"], **simulation_config)

    with ThreadPool(3) as pool:
        for result in pool.map(func, result_list):
            stats_list_result.append(result)

    return _delete_duplicates_from_result(stats_list_result)


def simulate_alpha_list_multi(
    s: SingleSession,
    alpha_list: list,
    limit_of_concurrent_simulations: int = 3,
    limit_of_multi_simulations: int = 3,
    simulation_config: dict = DEFAULT_CONFIG,
    tags: Optional[list[str]] = None,
) -> list:
    """
    Simulate a list of alphas using multi-simulation when possible.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_list (list): A list of alpha configurations to simulate.
        limit_of_concurrent_simulations (int, optional): The maximum number of concurrent simulation batches. Defaults to 3.
        limit_of_multi_simulations (int, optional): The maximum number of alphas in a multi-simulation. Defaults to 3.
        simulation_config (dict, optional): Configuration for the simulation. Defaults to DEFAULT_CONFIG.

    Returns:
        list: A list of dictionaries containing simulation results for each alpha.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API requests.
    """
    if (limit_of_multi_simulations < 2) or (limit_of_multi_simulations > 10):
        logger.warning("Limit of multi-simulation should be 2..10, will be set to 3")
        limit_of_multi_simulations = 3
    if (limit_of_concurrent_simulations < 1) or (limit_of_concurrent_simulations > 8):
        logger.warning("Limit of concurrent simulation should be 1..8, will be set to 3")
        limit_of_concurrent_simulations = 3
    if len(alpha_list) < 10:
        logger.warning(
            "List of alphas too short, single concurrent simulations will be used instead of multisimulations"
        )
        return simulate_alpha_list(
            s,
            alpha_list,
            limit_of_concurrent_simulations=limit_of_concurrent_simulations,
            simulation_config=simulation_config,
        )
    if any(d["type"] == "SUPER" for d in alpha_list):
        logger.warning("Multi-Simulation is not supported for SuperAlphas, single concurrent simulations will be used")
        return simulate_alpha_list(
            s,
            alpha_list,
            limit_of_concurrent_simulations=3,
            simulation_config=simulation_config,
        )

    tasks = [
        alpha_list[i : i + limit_of_multi_simulations] for i in range(0, len(alpha_list), limit_of_multi_simulations)
    ]
    result_list = []

    with ThreadPool(limit_of_concurrent_simulations) as pool:
        with tqdm.tqdm(total=len(tasks)) as pbar:
            for result in pool.imap_unordered(partial(simulate_multi_alpha, s, tags=tags), tasks):
                result_list.append(result)
                pbar.update()
    result_list_flat = [item for sublist in result_list for item in sublist]

    stats_list_result = []

    def func(x):
        return get_specified_alpha_stats(s, x["alpha_id"], x["simulate_data"], **simulation_config)

    with ThreadPool(3) as pool:
        for result in pool.map(func, result_list_flat):
            stats_list_result.append(result)

    return _delete_duplicates_from_result(stats_list_result)


def _delete_duplicates_from_result(result: list) -> list:
    """
    Remove duplicate alpha results from the simulation output.

    Args:
        result (list): A list of dictionaries containing simulation results.

    Returns:
        list: A deduplicated list of simulation results.
    """
    alpha_id_lst = []
    result_new = []
    for x in result:
        if x["alpha_id"] is not None:
            if x["alpha_id"] not in alpha_id_lst:
                result_new.append(x)
                alpha_id_lst.append(x["alpha_id"])
        else:
            result_new.append(x)
    return result_new


def set_alpha_properties(
    s: SingleSession,
    alpha_id: str,
    name: Optional[str] = None,
    color: Optional[str] = None,
    regular_desc: Optional[str] = None,
    selection_desc: str = "None",
    combo_desc: str = "None",
    tags: Optional[list[str]] = None,
) -> requests.Response:
    """
    Update the properties of an alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha to update.
        name (str, optional): The new name for the alpha. Defaults to None.
        color (str, optional): The new color for the alpha. Defaults to None.
        regular_desc (str, optional): Description for regular alpha. Defaults to None.
        selection_desc (str, optional): Description for the selection part of a super alpha. Defaults to "None".
        combo_desc (str, optional): Description for the combo part of a super alpha. Defaults to "None".
        tags (list, optional): List of tags to apply to the alpha. Defaults to None.

    Returns:
        requests.Response: The response object from the API call.
    """

    params = {}
    if name is not None:
        params["name"] = name
    if color is not None:
        params["color"] = color
    if tags is not None:
        params["tags"] = tags
    if regular_desc is not None:
        params.setdefault("regular", {})["description"] = regular_desc
    if selection_desc != "None":  # Assuming "None" is the default string value for selection_desc
        params.setdefault("selection", {})["description"] = selection_desc
    if combo_desc != "None":  # Assuming "None" is the default string value for combo_desc
        params.setdefault("combo", {})["description"] = combo_desc
    
    response = s.patch(brain_api_url + "/alphas/" + alpha_id, json=params)

    return response


def _get_alpha_pnl(
    s: SingleSession,
    alpha_id: str,
    pnl_type: str = "pnl",
) -> pd.DataFrame:
    """
    Retrieve the PnL data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.
        pnl_type (str): 'pnl' to get cumulative pnl, 'daily-pnl' to get daily pnl.

    Returns:
        pandas.DataFrame: A DataFrame containing the PnL data for the alpha.
    """

    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + f"/recordsets/{pnl_type}")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    pnl = result.json()
    if pnl.get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in pnl["schema"]["properties"]]
    pnl_df = (
        pd.DataFrame(pnl["records"], columns=columns)
        .assign(alpha_id=alpha_id, date=lambda x: pd.to_datetime(x.date, format="%Y-%m-%d"))
        .set_index("date")
    )
    return pnl_df


def get_alpha_pnl(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the cumulative PnL data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the PnL data for the alpha.
    """

    return _get_alpha_pnl(s, alpha_id, "pnl")


def get_alpha_yearly_stats(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the yearly statistics for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the yearly statistics for the alpha.
    """

    while True:
        result = s.get(brain_api_url + "/alphas/" + alpha_id + "/recordsets/yearly-stats")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    stats = result.json()

    if stats.get("records", 0) == 0:
        return pd.DataFrame()
    columns = [dct["name"] for dct in stats["schema"]["properties"]]
    yearly_stats_df = pd.DataFrame(stats["records"], columns=columns).assign(alpha_id=alpha_id)
    return yearly_stats_df


def get_datasets(
    s: SingleSession,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: str = "ALL",
) -> pd.DataFrame:
    """
    Retrieve available datasets based on specified parameters.

    Args:
        s (SingleSession): An authenticated session object.
        instrument_type (str, optional): The type of instrument. Defaults to "EQUITY".
        region (str, optional): The region. Defaults to "USA".
        delay (int, optional): The delay. Defaults to 1.
        universe (str, optional): The universe. Defaults to "TOP3000".
        theme (str, optional): The theme. Defaults to "ALL".

    Returns:
        pandas.DataFrame: A DataFrame containing information about available datasets.
    """
    if theme == "ALL":
        # Fetch both theme=false and theme=true
        url_false = (
            brain_api_url
            + "/data-sets?"
            + f"instrumentType={instrument_type}&region={region}&delay={str(delay)}&universe={universe}&theme=false"
        )
        result_false = s.get(url_false)
        df_false = pd.DataFrame(result_false.json()["results"])
        
        url_true = (
            brain_api_url
            + "/data-sets?"
            + f"instrumentType={instrument_type}&region={region}&delay={str(delay)}&universe={universe}&theme=true"
        )
        result_true = s.get(url_true)
        df_true = pd.DataFrame(result_true.json()["results"])
        
        datasets_df = pd.concat([df_false, df_true], ignore_index=True)
    else:
        url = (
            brain_api_url
            + "/data-sets?"
            + f"instrumentType={instrument_type}&region={region}&delay={str(delay)}&universe={universe}&theme={theme}"
        )
        result = s.get(url)
        datasets_df = pd.DataFrame(result.json()["results"])
        
    datasets_df = expand_dict_columns(datasets_df)
    return datasets_df


def get_datafields(
    s: SingleSession,
    instrument_type: str = "EQUITY",
    region: str = "USA",
    delay: int = 1,
    universe: str = "TOP3000",
    theme: str = "false",
    dataset_id: str = "",
    data_type: str = "MATRIX",
    search: str = "",
) -> pd.DataFrame:
    """
    Retrieve available datafields based on specified parameters.

    Args:
        s (SingleSession): An authenticated session object.
        instrument_type (str, optional): The type of instrument. Defaults to "EQUITY".
        region (str, optional): The region. Defaults to "USA".
        delay (int, optional): The delay. Defaults to 1.
        universe (str, optional): The universe. Defaults to "TOP3000".
        theme (str, optional): The theme. Defaults to "false".
        dataset_id (str, optional): The ID of a specific dataset. Defaults to "".
        data_type (str, optional): The type of data. Defaults to "MATRIX".
        search (str, optional): A search string to filter datafields. Defaults to "".

    Returns:
        pandas.DataFrame: A DataFrame containing information about available datafields.
    """
    type_param = f"&type={data_type}" if data_type != "ALL" else ""
    if len(search) == 0:
        url_template = (
            brain_api_url
            + "/data-fields?"
            + f"&instrumentType={instrument_type}"
            + f"&region={region}&delay={str(delay)}&universe={universe}{type_param}&dataset.id={dataset_id}&limit=50"
            + "&offset={x}"
        )
        count = s.get(url_template.format(x=0)).json()["count"]
        if count == 0:
            logger.warning(
                f"No fields found: region={region}, delay={str(delay)}, universe={universe}, "
                f"type={data_type}, dataset.id={dataset_id}"
            )
            return pd.DataFrame()

    else:
        url_template = (
            brain_api_url
            + "/data-fields?"
            + f"&instrumentType={instrument_type}"
            + f"&region={region}&delay={str(delay)}&universe={universe}{type_param}&limit=50"
            + f"&search={search}"
            + "&offset={x}"
        )
        count = 100

    max_try = 5
    datafields_list = []
    for x in range(0, count, 50):
        for _ in range(max_try):
            datafields = s.get(url_template.format(x=x))
            while datafields.status_code == 429:
                print("status_code 429, sleep 3 seconds")
                time.sleep(3)
                datafields = s.get(url_template.format(x=x))
            if "results" in datafields.json():
                break
            time.sleep(5)

        datafields_list.append(datafields.json()["results"])

    datafields_list_flat = [item for sublist in datafields_list for item in sublist]

    datafields_df = pd.DataFrame(datafields_list_flat)
    datafields_df = expand_dict_columns(datafields_df)
    return datafields_df


def get_operators(s: SingleSession) -> pd.DataFrame:
    """
    Fetches and processes the list of operators from the WorldQuant Brain API.

    This function retrieves the operators from the provided session `s`,
    explodes the 'scope' column (which contains lists) into separate rows,
    and returns the resulting DataFrame.

    Args:
    s (SingleSession): An authenticated session object.

    Returns:
    pd.DataFrame: A DataFrame containing the operators with each scope entry
    as a separate row.
    """
    df = pd.DataFrame(s.get(brain_api_url + "/operators").json())
    return df.explode('scope').reset_index(drop=True)


def get_instrument_type_region_delay(s: SingleSession) -> pd.DataFrame:
    """
    Retrieves and organizes instrument type, region, and delay data into a DataFrame.

    Parameters:
        s (SingleSession): The session object used for making the API call.

    Returns:
        df (pd.DataFrame): A DataFrame containing the instrument type, region, delay, universe, and neutralization data.

    The function fetches the settings options from the simulations endpoint and extracts the 'Instrument type',
    'Region', 'Universe', 'Delay', and 'Neutralization' data. It then organizes this data into a list of dictionaries,
    each containing the instrument type, region, delay, universe, and neutralization for a particular combination
    of instrument type, region, and delay. This list is then converted into a DataFrame and returned.
    """

    settings_options = s.options('https://api.worldquantbrain.com/simulations').json()['actions']['POST']['settings'][
        'children'
    ]
    data = [
        {settings_options[key]['label']: settings_options[key]['choices']}
        for key in settings_options.keys()
        if settings_options[key]['type'] == 'choice'
    ]

    instrument_type_data = {}
    region_data = {}
    universe_data = {}
    delay_data = {}
    neutralization_data = {}

    for item in data:
        if 'Instrument type' in item:
            instrument_type_data = item['Instrument type']
        elif 'Region' in item:
            region_data = item['Region']['instrumentType']
        elif 'Universe' in item:
            universe_data = item['Universe']['instrumentType']
        elif 'Delay' in item:
            delay_data = item['Delay']['instrumentType']
        elif 'Neutralization' in item:
            neutralization_data = item['Neutralization']['instrumentType']

    data_list = []

    for instrument_type in instrument_type_data:
        for region in region_data[instrument_type['value']]:
            for delay in delay_data[instrument_type['value']]['region'][region['value']]:
                row = {'InstrumentType': instrument_type['value'], 'Region': region['value'], 'Delay': delay['value']}
                row['Universe'] = [
                    item['value'] for item in universe_data[instrument_type['value']]['region'][region['value']]
                ]
                row['Neutralization'] = [
                    item['value'] for item in neutralization_data[instrument_type['value']]['region'][region['value']]
                ]
                data_list.append(row)

    df = pd.DataFrame(data_list).sort_values(
        by=['InstrumentType', 'Region', 'Delay'], ascending=False, ignore_index=True
    )
    return df


def performance_comparison(
    s: SingleSession, alpha_id: str, team_id: Optional[str] = None, competition: Optional[str] = None
) -> dict:
    """
    Retrieve performance comparison data for merged performance check.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.
        team_id (str, optional): The ID of the team for comparison. Defaults to None.
        competition (str, optional): The ID of the competition for comparison. Defaults to None.

    Returns:
        dict: A dictionary containing the performance comparison data.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """
    if competition is not None:
        part_url = f"competitions/{competition}"
    elif team_id is not None:
        part_url = f"teams/{team_id}"
    else:
        part_url = "users/self"
    while True:
        result = s.get(brain_api_url + f"/{part_url}/alphas/" + alpha_id + "/before-and-after-performance")
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    if result.json().get("stats", 0) == 0:
        logger.warning(f"Cant get performance comparison for alpha_id {alpha_id}. {result.json()}")
        return {}
    if result.status_code != 200:
        logger.warning(f"Cant get performance comparison for alpha_id {alpha_id}. {result.json()}")
        return {}

    return result.json()


def construct_selection_expression(
    selection: str,
    instrument_type: Literal["EQUITY"] = "EQUITY",
    region: str = "USA",
    delay: Literal[0, 1] = 1,
    selection_limit: int = 1000,
    selection_handling: str = "POSITIVE",
) -> dict:
    """
    Construct a dictionary containing parameters for a selection expression.

    This function creates a dictionary with the necessary parameters to define
    a selection expression for use in super alpha simulations.

    Args:
        selection (str): The selection expression to be used.
        instrument_type (str, optional): Instrument type to use.
            Defaults to "EQUITY".
        region (str, optional): The geographic region for the selection.
            Defaults to "USA".
        delay (int, optional): The delay parameter for the selection.
            Defaults to 1.
        selection_limit (int, optional): The maximum number of instruments
            to be selected. Defaults to 1000.
        selection_handling (str, optional): The method for handling the
            selection. Defaults to "POSITIVE".

    Returns:
        dict: A dictionary containing the constructed selection expression
        parameters, ready to be used in API calls or other functions.
    """
    selection_data = {
        "settings.instrumentType": instrument_type,
        "settings.region": region,
        "settings.delay": delay,
        "selection": selection,
        "limit": 10,
        "selectionLimit": selection_limit,
        "selectionHandling": selection_handling,
    }
    return selection_data


def run_selection(s: SingleSession, selection_data: dict) -> dict:
    """
    Run a selection simulation using the provided selection data.

    Args:
        s (SingleSession): An authenticated session object.
        selection_data (dict): A dictionary containing the selection parameters.

    Returns:
        dict: A dictionary containing the count of selected alphas and any messages.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """
    selection_response = s.get(brain_api_url + "/simulations/super-selection", params=selection_data)
    r = selection_response.json()
    selected_alphas_count = r.get("count")
    message = r.get("message", "")
    time.sleep(2)
    return {"selected_alphas_count": selected_alphas_count, "message": message}


def get_alpha_daily_pnl(s: SingleSession, alpha_id: str) -> pd.DataFrame:
    """
    Retrieve the daily PnL data for a specific alpha.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha.

    Returns:
        pandas.DataFrame: A DataFrame containing the PnL data for the alpha.
    """

    return _get_alpha_pnl(s, alpha_id, "daily-pnl")


def submit_alpha(s: SingleSession, alpha_id: str) -> bool:
    """
    Submit an alpha for evaluation.

    Args:
        s (SingleSession): An authenticated session object.
        alpha_id (str): The ID of the alpha to submit.

    Returns:
        bool: True if the submission was successful, False otherwise.

    Raises:
        requests.exceptions.RequestException: If there's an error in the API request.
    """
    result = s.post(brain_api_url + "/alphas/" + alpha_id + "/submit")
    while True:
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
            result = s.get(brain_api_url + "/alphas/" + alpha_id + "/submit")
        else:
            break
    return result.status_code == 200



def main():
    """
    Main function to demonstrate the usage of the library.

    This function creates a session, generates a list of sample alphas,
    and simulates them using the simulate_alpha_list function.
    """

    s = start_session()

    k = [
        "vwap * 2",
        "open * close",
        "high * low",
        "vwap * 3",
        "open * close",
        "high * low",
    ]
    alpha_list = [generate_alpha(x) for x in k]

    simulate_alpha_list(s, alpha_list)


if __name__ == "__main__":
    main()
