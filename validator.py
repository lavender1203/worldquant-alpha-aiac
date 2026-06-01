#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表达式验证器 - 使用抽象语法树验证字符串表达式格式是否正确

本模块实现了一个能够检测字符串表达式格式是否正确的系统，基于PLY(Python Lex-Yacc)
构建词法分析器和语法分析器，识别表达式中的操作符、函数和字段，并验证其格式正确性。
"""

import re
import sys
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# 尝试导入PLY库，如果不存在则提供安装提示
try:
    import ply.lex as lex
    import ply.yacc as yacc
except ImportError:
    print("错误: 需要安装PLY库。请运行 'pip install ply' 来安装。")
    sys.exit(1)

# 1. 定义支持的操作符和函数
supported_functions = {
    # Group 类别函数
    'group_min': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_mean': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression']},
    'group_median': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_max': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_rank': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_vector_proj': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'category']},
    'group_normalize': {'min_args': 2, 'max_args': 5, 'arg_types': ['expression', 'category', 'expression', 'expression', 'expression']},
    'group_extra': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'category']},
    'group_backfill': {'min_args': 3, 'max_args': 4, 'arg_types': ['expression', 'expression', 'expression', 'expression'], 'param_names': ['x', 'cat', 'days', 'std']},
    'group_scale': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_count': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_zscore': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_std_dev': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_sum': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_neutralize': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'category']},
    'group_multi_regression': {'min_args': 4, 'max_args': 9, 'arg_types': ['expression'] * 9},
    'group_cartesian_product': {'min_args': 2, 'max_args': 2, 'arg_types': ['category', 'category']},
    'combo_a': {'min_args': 1, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression']},
    
    # Transformational 类别函数
    'right_tail': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'bucket': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression']},  # 第二个参数可以是string类型的range参数
    'tail': {'min_args': 3, 'max_args': 4, 'arg_types': ['expression', 'expression', 'expression', 'expression'], 'param_names': ['x', 'lower', 'upper', 'newval']},
    'left_tail': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'trade_when': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression']},
    'generate_stats': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    
    # Cross Sectional 类别函数
    'winsorize': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['x', 'std']},
    'rank': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'regression_proj': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'vector_neut': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'regression_neut': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'multi_regression': {'min_args': 2, 'max_args': 100, 'arg_types': ['expression'] * 100},  # 支持多个自变量
    
    # Time Series 类别函数
    'ts_std_dev': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_mean': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_delay': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_corr': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number']},
    'ts_zscore': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_returns': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'number'], 'param_names': ['x', 'd', 'mode']},
    'ts_product': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_backfill': {'min_args': 2, 'max_args': 4, 'arg_types': ['expression', 'number', 'number', 'string']},
    'days_from_last_change': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'last_diff_value': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_scale': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'number']},
    'ts_entropy': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number'], 'param_names': ['x', 'd', 'buckets']},
    'ts_step': {'min_args': 1, 'max_args': 1, 'arg_types': ['number']},
    'ts_sum': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_co_kurtosis': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number']},
    'inst_tvr': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_decay_exp_window': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'number'], 'param_names': ['x', 'd', 'factor']},
    'ts_av_diff': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_kurtosis': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_min_max_diff': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'number']},
    'ts_arg_max': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_max': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_min_max_cps': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'number']},
    'ts_rank': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'number']},
    'ts_ir': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_theilsen': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number']},
    'hump_decay': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_weighted_decay': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_quantile': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'string']},
    'ts_min': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_count_nans': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_covariance': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number']},
    'ts_co_skewness': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number']},
    'ts_min_diff': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_decay_linear': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'boolean']},
    'jump_decay': {'min_args': 2, 'max_args': 5, 'arg_types': ['expression', 'number', 'expression', 'number', 'number'], 'param_names': ['x', 'd', 'stddev', 'sensitivity', 'force']},
    'ts_moment': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'number', 'number'], 'param_names': ['x', 'd', 'k']},
    'ts_arg_min': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_regression': {'min_args': 3, 'max_args': 5, 'arg_types': ['expression', 'expression', 'number', 'number', 'number'], 'param_names': ['y', 'x', 'd', 'lag', 'rettype']},
    'ts_skewness': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_max_diff': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'kth_element': {'min_args': 3, 'max_args': 4, 'arg_types': ['expression', 'number', 'number', 'string'], 'param_names': ['x', 'd', 'k', 'ignore']},
    'hump': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'number'], 'param_names': ['x', 'hump']},
    'ts_median': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_delta': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_poly_regression': {'min_args': 3, 'max_args': 4, 'arg_types': ['expression', 'expression', 'number', 'number']},
    'ts_target_tvr_decay': {'min_args': 1, 'max_args': 4, 'arg_types': ['expression', 'number', 'number', 'number'], 'param_names': ['x', 'lambda_min', 'lambda_max', 'target_tvr']},
    'ts_target_tvr_delta_limit': {'min_args': 2, 'max_args': 5, 'arg_types': ['expression', 'expression', 'number', 'number', 'number'], 'param_names': ['x', 'y', 'lambda_min', 'lambda_max', 'target_tvr']},
    'ts_target_tvr_hump': {'min_args': 1, 'max_args': 4, 'arg_types': ['expression', 'number', 'number', 'number'], 'param_names': ['x', 'lambda_min', 'lambda_max', 'target_tvr']},
    'ts_delta_limit': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number']},
    
    # Special 类别函数
    'inst_pnl': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'self_corr': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'in': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},  # 注意：这是关键字
    'universe_size': {'min_args': 0, 'max_args': 0, 'arg_types': []},
    
    # Missing functions from operators.py
    'quantile': {'min_args': 1, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression'], 'param_names': ['x', 'driver', 'sigma']},  # quantile(x, driver = gaussian, sigma = 1.0)
    'normalize': {'min_args': 1, 'max_args': 3, 'arg_types': ['expression', 'boolean', 'number']},  # normalize(x, useStd = false, limit = 0.0)
    'zscore': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},  # zscore(x)
    
    # Logical 类别函数
    'or': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},  # 注意：这是关键字
    'and': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},  # 注意：这是关键字
    'not': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},  # 注意：这是关键字
    'is_nan': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'is_not_nan': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'less': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'equal': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'greater': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'is_finite': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'if_else': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression']},
    'not_equal': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'less_equal': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'greater_equal': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    
    # Vector 类别函数
    'vec_kurtosis': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_min': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_count': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_sum': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_skewness': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_max': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_avg': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_range': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_choose': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number'], 'param_names': ['x', 'nth']},
    'vec_powersum': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'number'], 'param_names': ['x', 'constant']},
    'vec_stddev': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_percentage': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'number'], 'param_names': ['x', 'percentage']},
    'vec_ir': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'vec_norm': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'ts_percentage': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'number', 'number'], 'param_names': ['x', 'd', 'percentage']},
    'signed_power': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    'ts_product': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'number']},
    
    # Additional functions from test cases
    'rank_by_side': {'min_args': 1, 'max_args': 3, 'arg_types': ['expression', 'number', 'number'], 'param_names': ['x', 'rate', 'scale']},
    'log_diff': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'nan_mask': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},
    'ts_partial_corr': {'min_args': 4, 'max_args': 4, 'arg_types': ['expression', 'expression', 'expression', 'number']},
    'ts_triple_corr': {'min_args': 4, 'max_args': 4, 'arg_types': ['expression', 'expression', 'expression', 'number']},
    'clamp': {'min_args': 1, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression'], 'param_names': ['x', 'lower', 'upper']},
    'keep': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number'], 'param_names': ['x', 'condition', 'period']},
    'replace': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression'], 'param_names': ['x', 'target', 'dest']},
    'filter': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression'], 'param_names': ['x', 'h', 't']},
    'one_side': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'string'], 'param_names': ['x', 'side']},
    'scale_down': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['x']},
    
    # Arithmetic 类别函数
    'add': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'expression', 'boolean']},  # add(x, y, filter=false)
    'multiply': {'min_args': 2, 'max_args': 100, 'arg_types': ['expression'] * 99 + ['boolean'], 'param_names': ['x', 'y', 'filter']},  # multiply(x, y, ..., filter=false)
    'sign': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'subtract': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'expression', 'boolean']},  # subtract(x, y, filter=false)
    'pasteurize': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'log': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'purify': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'arc_tan': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'max': {'min_args': 2, 'max_args': 100, 'arg_types': ['expression'] * 100},  # max(x, y, ...)
    'to_nan': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['x']},
    'abs': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'sigmoid': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'divide': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},  # divide(x, y)
    'min': {'min_args': 2, 'max_args': 100, 'arg_types': ['expression'] * 100},  # min(x, y, ...)
    'tanh': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'nan_out': {'min_args': 1, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression'], 'param_names': ['x', 'lower', 'upper']},  # nan_out(x, lower=0, upper=0)
    'signed_power': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},  # signed_power(x, y)
    'inverse': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'round': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'sqrt': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    's_log_1p': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'reverse': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},  # -x
    'power': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression']},  # power(x, y)
    'densify': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    'floor': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression']},
    # Appended missing operators
    'arc_cos': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['x']},
    'arc_sin': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['x']},
    'ceiling': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['x']},
    'exp': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['x']},
    'fraction': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['x']},
    'round_down': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['x']},
    'is_not_finite': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'negate': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'ts_rank_gmean_amean_diff': {'min_args': 5, 'max_args': 5, 'arg_types': ['expression', 'expression', 'expression', 'expression', 'number'], 'param_names': ['input1', 'input2', 'input3', '...', 'd']},
    'ts_vector_neut': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number'], 'param_names': ['x', 'y', 'd']},
    'ts_vector_proj': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'number'], 'param_names': ['x', 'y', 'd']},
    'scale': {'min_args': 1, 'max_args': 4, 'arg_types': ['expression', 'expression', 'expression', 'expression'], 'param_names': ['x', 'scale', 'longscale', 'shortscale']},
    'generalized_rank': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['open', 'm']},
    'rank_gmean_amean_diff': {'min_args': 4, 'max_args': 4, 'arg_types': ['expression', 'expression', 'expression', 'expression'], 'param_names': ['input1', 'input2', 'input3', '...']},
    'truncate': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['x', 'maxPercent']},
    'vector_proj': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['x', 'y']},
    'vec_filter': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['vec', 'value']},
    'group_coalesce': {'min_args': 4, 'max_args': 4, 'arg_types': ['expression', 'expression', 'expression', 'expression'], 'param_names': ['original_group', 'group2', 'group3', '…']},
    'group_percentage': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'category', 'expression'], 'param_names': ['x', 'group', 'percentage']},
    'group_vector_neut': {'min_args': 3, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression'], 'param_names': ['x', 'y', 'g']},
    'convert': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['x', 'mode']},
    'reduce_avg': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['input', 'threshold']},
    'reduce_choose': {'min_args': 2, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression'], 'param_names': ['input', 'nth', 'ignoreNan']},
    'reduce_count': {'min_args': 2, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['input', 'threshold']},
    'reduce_ir': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'reduce_kurtosis': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'reduce_max': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'reduce_min': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'reduce_norm': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'reduce_percentage': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['input', 'percentage']},
    'reduce_powersum': {'min_args': 1, 'max_args': 3, 'arg_types': ['expression', 'expression', 'expression'], 'param_names': ['input', 'constant', 'precise']},
    'reduce_range': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'reduce_skewness': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
    'reduce_stddev': {'min_args': 1, 'max_args': 2, 'arg_types': ['expression', 'expression'], 'param_names': ['input', 'threshold']},
    'reduce_sum': {'min_args': 1, 'max_args': 1, 'arg_types': ['expression'], 'param_names': ['input']},
}

# 2. 定义group类型字段
group_fields = {
    'sector', 'subindustry', 'industry', 'exchange', 'country', 'market'
}

# 3. 有效类别集合
valid_categories = group_fields

# 4. 字段命名模式 - 只校验字段是不是数字字母下划线组成
field_patterns = [
    re.compile(r'^[a-zA-Z0-9_]+$'),  # 只允许数字、字母和下划线组成的字段名
]

# 4. 抽象语法树节点类型
class ASTNode:
    """抽象语法树节点基类"""
    def __init__(self, node_type: str, children: Optional[List['ASTNode']] = None, 
                 value: Optional[Any] = None, line: Optional[int] = None):
        self.node_type = node_type  # 'function', 'operator', 'field', 'number', 'expression'
        self.children = children or []
        self.value = value
        self.line = line
    
    def __str__(self) -> str:
        return f"ASTNode({self.node_type}, {self.value}, line={self.line})"
    
    def __repr__(self) -> str:
        return self.__str__()

class ExpressionValidator:
    """表达式验证器类"""
    
    def __init__(self):
        """初始化词法分析器和语法分析器"""
        # 构建词法分析器
        self.lexer = lex.lex(module=self, debug=False)
        # 构建语法分析器
        self.parser = yacc.yacc(module=self, debug=False)
        # 错误信息存储
        self.errors = []
    
    # 词法分析器规则
    tokens = ('FUNCTION', 'FIELD', 'NUMBER', 'LPAREN', 'RPAREN', 
              'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'COMMA', 'CATEGORY',
              'EQUAL', 'ASSIGN', 'IDENTIFIER', 'STRING', 'GREATER', 'LESS', 'GREATEREQUAL', 'LESSEQUAL', 'NOTEQUAL', 'BOOLEAN')
    
    # 忽略空白字符
    t_ignore = ' \t\n'
    
    # 操作符 - 注意顺序很重要，长的操作符要放在前面
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_COMMA = r','    
    t_EQUAL = r'=='
    t_NOTEQUAL = r'!='
    t_GREATEREQUAL = r'>='
    t_LESSEQUAL = r'<='
    t_GREATER = r'>'
    t_LESS = r'<'
    t_ASSIGN = r'='
    
    # 数字（整数和浮点数）
    def t_NUMBER(self, t):
        r'\d+\.?\d*'
        if '.' in t.value:
            t.value = float(t.value)
        else:
            t.value = int(t.value)
        return t
    
    # 字符串 - 需要放在所有其他标识符规则之前
    def t_STRING(self, t):
        r"'[^']*'|\"[^\"]*\""
        # 去除引号
        t.value = t.value[1:-1]
        return t
    
    # 函数和字段名
    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_][a-zA-Z0-9_]*'
        # 检查是否为布尔值
        if t.value.lower() in {'true', 'false'}:
            t.type = 'BOOLEAN'
            t.value = t.value.lower()  # 转换为小写以保持一致性
        else:
            # 查看当前token后面的字符，判断是否为参数名（后面跟着'='）
            lexpos = t.lexpos
            next_chars = ''
            if lexpos + len(t.value) < len(t.lexer.lexdata):
                # 查看当前token后面的字符，跳过空格
                next_pos = lexpos + len(t.value)
                while next_pos < len(t.lexer.lexdata) and t.lexer.lexdata[next_pos].isspace():
                    next_pos += 1
                if next_pos < len(t.lexer.lexdata):
                    next_chars = t.lexer.lexdata[next_pos:next_pos+1]
            
            # 如果后面跟着'='，则为参数名
            if next_chars == '=':
                t.type = 'IDENTIFIER'
            # 如果后面跟着'('，则为函数名
            elif next_chars == '(':
                t.type = 'FUNCTION'
                t.value = t.value.lower()  # 转换为小写以保持一致性
            # 检查是否为参数名（支持更多参数名）
            elif t.value in {'std', 'k', 'ignore', 'lambda_min', 'lambda_max', 'target_tvr', 'range', 'buckets', 'lag', 'rettype', 'mode', 'nth', 'constant', 'percentage', 'driver', 'sigma', 'rate', 'scale', 'filter', 'lower', 'upper', 'target', 'dest', 'event', 'sensitivity', 'force', 'h', 't', 'period', 'stddev', 'factor', 'k', 'useStd', 'limit', 'gaussian', 'uniform', 'cauchy', 'value', 'reverse'}:
                t.type = 'IDENTIFIER'
            # 检查是否为函数名（不区分大小写）
            elif t.value.lower() in supported_functions:
                t.type = 'FUNCTION'
                t.value = t.value.lower()  # 转换为小写以保持一致性
            # 检查是否为有效类别
            elif t.value in valid_categories:
                t.type = 'CATEGORY'
            # 检查是否为字段名
            elif self._is_valid_field(t.value):
                t.type = 'FIELD'
            else:
                # 其他标识符，保留为IDENTIFIER类型
                t.type = 'IDENTIFIER'
        return t
    
    # 行号跟踪
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    # 错误处理
    def t_error(self, t):
        if t:
            # 检查是否为非法字符
            if not re.match(r'[a-zA-Z0-9_\+\-\*/\(\)\,\s=<>!]', t.value[0]):
                # 这是一个非法字符
                self.errors.append(f"非法字符 '{t.value[0]}' (行 {t.lexer.lineno})")
            else:
                # 这是一个非法标记
                self.errors.append(f"非法标记 '{t.value}' (行 {t.lexer.lineno})")
            # 跳过这个字符，继续处理
            t.lexer.skip(1)
        else:
            self.errors.append("词法分析器到达文件末尾")
    
    # 语法分析器规则
    def p_expression(self, p):
        """expression : comparison
                      | expression EQUAL comparison
                      | expression NOTEQUAL comparison
                      | expression GREATER comparison
                      | expression LESS comparison
                      | expression GREATEREQUAL comparison
                      | expression LESSEQUAL comparison"""
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ASTNode('binop', [p[1], p[3]], {'op': p[2]})
    
    def p_comparison(self, p):
        """comparison : term
                      | comparison PLUS term
                      | comparison MINUS term"""
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ASTNode('binop', [p[1], p[3]], {'op': p[2]})
    
    def p_term(self, p):
        """term : factor
                | term TIMES factor
                | term DIVIDE factor"""
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = ASTNode('binop', [p[1], p[3]], {'op': p[2]})
    
    def p_factor(self, p):
        """factor : NUMBER
                  | STRING
                  | FIELD
                  | CATEGORY
                  | IDENTIFIER
                  | BOOLEAN
                  | MINUS factor
                  | LPAREN expression RPAREN
                  | function_call"""
        if len(p) == 2:
            # 数字、字符串、字段、类别或标识符
            if p.slice[1].type == 'NUMBER':
                p[0] = ASTNode('number', value=p[1])
            elif p.slice[1].type == 'STRING':
                p[0] = ASTNode('string', value=p[1])
            elif p.slice[1].type == 'FIELD':
                p[0] = ASTNode('field', value=p[1])
            elif p.slice[1].type == 'CATEGORY':
                p[0] = ASTNode('category', value=p[1])
            elif p.slice[1].type == 'BOOLEAN':
                p[0] = ASTNode('boolean', value=p[1])
            elif p.slice[1].type == 'IDENTIFIER':
                p[0] = ASTNode('identifier', value=p[1])
            else:
                p[0] = p[1]
        elif len(p) == 3:
            # 一元负号
            p[0] = ASTNode('unop', [p[2]], {'op': p[1]})
        elif len(p) == 4:
            # 括号表达式
            p[0] = p[2]
        else:
            # 函数调用
            p[0] = p[1]
    
    def p_function_call(self, p):
        '''function_call : FUNCTION LPAREN args RPAREN'''
        p[0] = ASTNode('function', p[3], p[1])
    
    def p_args(self, p):
        '''args : arg_list
                | empty'''
        if len(p) == 2 and p[1] is not None:
            p[0] = p[1]
        else:
            p[0] = []
    
    def p_arg_list(self, p):
        '''arg_list : arg
                   | arg_list COMMA arg'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]
    
    def p_arg(self, p):
        '''arg : expression
              | IDENTIFIER ASSIGN expression'''
        if len(p) == 2:
            p[0] = {'type': 'positional', 'value': p[1]}
        else:
            p[0] = {'type': 'named', 'name': p[1], 'value': p[3]}
    
    def p_empty(self, p):
        '''empty :'''
        p[0] = None
    
    # 语法错误处理
    def p_error(self, p):
        if p:
            self.errors.append(f"语法错误在位置 {p.lexpos}: 非法标记 '{p.value}'")
        else:
            self.errors.append("语法错误: 表达式不完整")
    
    def _is_valid_field(self, field_name: str) -> bool:
        """检查字段名是否符合模式"""
        for pattern in field_patterns:
            if pattern.match(field_name):
                return True
        return False
    
    def validate_function(self, node: ASTNode, is_in_group_arg: bool = False, allowed_fields: Optional[List[str]] = None) -> List[str]:
        """验证函数调用的参数数量和类型"""
        function_name = node.value
        args = node.children
        function_info = supported_functions.get(function_name)
        
        if not function_info:
            return [f"未知函数: {function_name}"]
        
        errors = []
        
        # 检查参数数量
        if len(args) < function_info['min_args']:
            errors.append(f"函数 {function_name} 需要至少 {function_info['min_args']} 个参数，但只提供了 {len(args)}")
        elif len(args) > function_info['max_args']:
            errors.append(f"函数 {function_name} 最多接受 {function_info['max_args']} 个参数，但提供了 {len(args)}")
        
        # 处理参数验证
        # 跟踪已使用的位置参数索引
        positional_index = 0
        
        # 对于所有函数，支持命名参数
        for arg in args:
            if isinstance(arg, dict):
                if arg['type'] == 'named':
                    # 命名参数
                    if 'param_names' in function_info and arg['name'] in function_info['param_names']:
                        # 查找参数在param_names中的索引
                        param_index = function_info['param_names'].index(arg['name'])
                        if param_index < len(function_info['arg_types']):
                            expected_type = function_info['arg_types'][param_index]
                            arg_errors = self._validate_arg_type(arg['value'], expected_type, param_index, function_name, is_in_group_arg, allowed_fields)
                            errors.extend(arg_errors)
                    # 对于winsorize函数，支持std和clip参数
                    elif function_name == 'winsorize' and arg['name'] in ['std', 'clip']:
                        arg_errors = self._validate_arg_type(arg['value'], 'number', 0, function_name, is_in_group_arg, allowed_fields)
                        errors.extend(arg_errors)
                    # 对于bucket函数，支持'range'和'buckets'参数
                    elif function_name == 'bucket' and arg['name'] in ['range', 'buckets']:
                        # range和buckets参数应该是string类型
                        arg_errors = self._validate_arg_type(arg['value'], 'string', 1, function_name, is_in_group_arg, allowed_fields)
                        errors.extend(arg_errors)
                    else:
                        errors.append(f"函数 {function_name} 不存在参数 '{arg['name']}'")
                elif arg['type'] == 'positional':
                    # 位置参数（字典形式）
                    # 对于winsorize函数，第二个参数必须是命名参数
                    if function_name == 'winsorize' and positional_index == 1:
                        errors.append(f"函数 {function_name} 的第二个参数必须使用命名参数 'std='")
                    # 对于ts_moment函数，第三个参数必须是命名参数
                    elif function_name == 'ts_moment' and positional_index == 2:
                        errors.append(f"函数 {function_name} 的第三个参数必须使用命名参数 'k='")
                    else:
                        # 验证位置参数的类型
                        if positional_index < len(function_info['arg_types']):
                            expected_type = function_info['arg_types'][positional_index]
                            arg_errors = self._validate_arg_type(arg['value'], expected_type, positional_index, function_name, is_in_group_arg, allowed_fields)
                            errors.extend(arg_errors)
                    positional_index += 1
                else:
                    # 其他字典类型参数
                    errors.append(f"参数 {positional_index+1} 格式错误")
                    positional_index += 1
            else:
                # 位置参数（直接ASTNode形式）
                # 对于winsorize函数，第二个参数必须是命名参数
                if function_name == 'winsorize' and positional_index == 1:
                    errors.append(f"函数 {function_name} 的第二个参数必须使用命名参数 'std='")
                # 对于ts_moment函数，第三个参数必须是命名参数
                elif function_name == 'ts_moment' and positional_index == 2:
                    errors.append(f"函数 {function_name} 的第三个参数必须使用命名参数 'k='")
                else:
                    # 验证位置参数的类型
                    if positional_index < len(function_info['arg_types']):
                        expected_type = function_info['arg_types'][positional_index]
                        arg_errors = self._validate_arg_type(arg, expected_type, positional_index, function_name, is_in_group_arg, allowed_fields)
                        errors.extend(arg_errors)
                positional_index += 1
        
        return errors
    
    def _validate_arg_type(self, arg: ASTNode, expected_type: str, arg_index: int, function_name: str, is_in_group_arg: bool = False, allowed_fields: Optional[List[str]] = None) -> List[str]:
        """验证参数类型是否符合预期"""
        errors = []
        
        # 首先检查是否是group类型字段，如果是则只能用于Group类型函数
        # 但是如果当前函数是group_xxx或在group函数的参数链中，则允许使用
        if arg.node_type == 'category' and arg.value in group_fields:
            if not (function_name.startswith('group_') or is_in_group_arg):
                errors.append(f"Group类型字段 '{arg.value}' 只能用于Group类型函数的参数中")
        
        # 然后验证参数类型是否符合预期
        if expected_type == 'expression':
            # 表达式可以是任何有效的AST节点
            pass
        elif expected_type == 'number':
            if arg.node_type != 'number':
                errors.append(f"参数 {arg_index+1} 应该是一个数字，但得到 {arg.node_type}")
        elif expected_type == 'boolean':
            # 布尔值可以是数字（0/1）
            if arg.node_type != 'number':
                errors.append(f"参数 {arg_index+1} 应该是一个布尔值（0/1），但得到 {arg.node_type}")
        elif expected_type == 'field':
            if arg.node_type != 'field' and arg.node_type != 'category':
                # 允许field或category作为字段参数
                errors.append(f"参数 {arg_index+1} 应该是一个字段，但得到 {arg.node_type}")
            elif arg.node_type == 'field' and not self._is_valid_field(arg.value):
                errors.append(f"无效的字段名: {arg.value}")
            elif arg.node_type == 'field' and allowed_fields is not None:
                # 严格校验字段是否存在于 allowed_fields 中
                if arg.value not in allowed_fields:
                    errors.append(f"Field '{arg.value}' not found in dataset")

        elif expected_type == 'category':
            if not function_name.startswith('group_'):
                # 非group函数的category参数必须是category类型且在valid_categories中
                if arg.node_type != 'category':
                    errors.append(f"参数 {arg_index+1} 应该是一个类别，但得到 {arg.node_type}")
                elif arg.value not in valid_categories:
                    errors.append(f"无效的类别: {arg.value}")
            # group函数的category参数可以是任何类型（field、category等），不进行类型校验
        
        return errors
    
    def validate_ast(self, ast: Optional[ASTNode], is_in_group_arg: bool = False, allowed_fields: Optional[List[str]] = None) -> List[str]:
        """递归验证抽象语法树"""
        if not ast:
            return ["无法解析表达式"]
        
        errors = []
        
        # 根据节点类型进行验证
        if ast.node_type == 'function':
            # 检查当前函数是否是group函数
            is_group_function = ast.value.startswith('group_')
            # 确定当前是否在group函数的参数链中
            current_in_group_arg = is_in_group_arg or is_group_function
            # 验证函数
            function_errors = self.validate_function(ast, current_in_group_arg, allowed_fields)
            errors.extend(function_errors)
            
            # 递归验证子节点时使用current_in_group_arg
            for child in ast.children:
                if isinstance(child, dict):
                    # 命名参数，验证其值
                    if 'value' in child and hasattr(child['value'], 'node_type'):
                        child_errors = self.validate_ast(child['value'], current_in_group_arg, allowed_fields)
                        errors.extend(child_errors)
                elif hasattr(child, 'node_type'):
                    child_errors = self.validate_ast(child, current_in_group_arg, allowed_fields)
                    errors.extend(child_errors)
        elif ast.node_type in ['unop', 'binop']:
            # 对操作符的子节点进行验证
            for child in ast.children:
                if hasattr(child, 'node_type'):
                    child_errors = self.validate_ast(child, is_in_group_arg, allowed_fields)
                    errors.extend(child_errors)
        elif ast.node_type == 'field':
            # 验证字段名
            if not self._is_valid_field(ast.value):
                errors.append(f"无效的字段名: {ast.value}")
            elif allowed_fields is not None:
                # 严格校验字段是否存在于 allowed_fields 中
                if ast.value not in allowed_fields:
                    errors.append(f"Field '{ast.value}' not found in dataset")
        else:
            # 递归验证子节点
            for child in ast.children:
                if isinstance(child, dict):
                    # 命名参数，验证其值
                    if 'value' in child and hasattr(child['value'], 'node_type'):
                        child_errors = self.validate_ast(child['value'], is_in_group_arg, allowed_fields)
                        errors.extend(child_errors)
                elif hasattr(child, 'node_type'):
                    child_errors = self.validate_ast(child, is_in_group_arg, allowed_fields)
                    errors.extend(child_errors)
        
        return errors
    
    def _process_semicolon_expression(self, expression: str) -> Tuple[bool, str]:
        """处理带有分号的表达式，将其转换为不带分号的简化形式
        
        Args:
            expression: 要处理的表达式字符串
            
        Returns:
            Tuple[bool, str]: (是否成功, 转换后的表达式或错误信息)
        """
        # 检查表达式是否以分号结尾
        if expression.strip().endswith(';'):
            return False, "表达式不能以分号结尾"
        
        # 分割表达式为语句列表
        statements = [stmt.strip() for stmt in expression.split(';') if stmt.strip()]
        if not statements:
            return False, "表达式不能为空"
        
        # 存储变量赋值
        variables = {}
        
        # 处理每个赋值语句（除了最后一个）
        for i, stmt in enumerate(statements[:-1]):
            # 检查是否包含赋值符号
            if '=' not in stmt:
                return False, f"第{i+1}个语句必须是赋值语句（使用=符号）"
            
            # 检查是否是比较操作符（==, !=, <=, >=）
            if any(op in stmt for op in ['==', '!=', '<=', '>=']):
                # 如果包含比较操作符，需要确认是否有赋值符号
                # 使用临时替换法：将比较操作符替换为临时标记，再检查是否还有=
                temp_stmt = stmt
                for op in ['==', '!=', '<=', '>=']:
                    temp_stmt = temp_stmt.replace(op, '---')
                
                if '=' not in temp_stmt:
                    return False, f"第{i+1}个语句必须是赋值语句，不能只是比较表达式"
            
            # 找到第一个=符号（不是比较操作符的一部分）
            # 先将比较操作符替换为临时标记，再找=
            temp_stmt = stmt
            for op in ['==', '!=', '<=', '>=']:
                temp_stmt = temp_stmt.replace(op, '---')
            
            if '=' not in temp_stmt:
                return False, f"第{i+1}个语句必须是赋值语句（使用=符号）"
            
            # 找到实际的=位置
            equals_pos = temp_stmt.index('=')
            
            # 在原始语句中找到对应位置
            real_equals_pos = 0
            temp_count = 0
            for char in stmt:
                if temp_count == equals_pos:
                    break
                if char in '!<>':
                    # 检查是否是比较操作符的一部分
                    if real_equals_pos + 1 < len(stmt) and stmt[real_equals_pos + 1] == '=':
                        # 是比较操作符，跳过两个字符
                        real_equals_pos += 2
                        temp_count += 3  # 因为替换成了三个字符的---
                    else:
                        real_equals_pos += 1
                        temp_count += 1
                else:
                    real_equals_pos += 1
                    temp_count += 1
            
            # 分割变量名和值
            var_name = stmt[:real_equals_pos].strip()
            var_value = stmt[real_equals_pos + 1:].strip()
            
            # 检查变量名是否有效
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var_name):
                return False, f"第{i+1}个语句的变量名'{var_name}'无效，只能包含字母、数字和下划线，且不能以数字开头"
            
            var_name_lower = var_name.lower()  # 变量名不区分大小写
            
            # 检查变量名是否在后续表达式中使用
            # 这里不需要，因为后面的表达式会检查
            
            # 检查变量值中使用的变量是否已经定义
            # 简单检查：提取所有可能的变量名
            used_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', var_value)
            for used_var in used_vars:
                used_var_lower = used_var.lower()
                if used_var_lower not in variables:
                    # 检查是否是函数名
                    if used_var not in supported_functions:
                        # 对于单个字母或简单单词，不自动视为字段名，要求先定义
                        if len(used_var) <= 2:
                            return False, f"第{i+1}个语句中使用的变量'{used_var}'未在之前定义"
                        # 对于较长的字段名，仍然允许作为字段名
                        elif not self._is_valid_field(used_var):
                            return False, f"第{i+1}个语句中使用的变量'{used_var}'未在之前定义"
            
            # 将之前定义的变量替换到当前值中
            for existing_var, existing_val in variables.items():
                # 使用单词边界匹配，避免替换到其他单词的一部分
                var_value = re.sub(rf'\b{existing_var}\b', existing_val, var_value)
            
            # 存储变量
            variables[var_name_lower] = var_value
        
        # 处理最后一个语句（实际的表达式）
        final_stmt = statements[-1]
        
        # 检查最后一个语句是否是赋值语句
        if '=' in final_stmt:
            # 替换比较操作符为临时标记，然后检查是否还有单独的=
            temp_stmt = final_stmt
            for op in ['==', '!=', '<=', '>=']:
                temp_stmt = temp_stmt.replace(op, '---')
            
            if '=' in temp_stmt:
                return False, "最后一个语句不能是赋值语句"
        
        # 检查最后一个语句中使用的变量是否已经定义
        used_vars = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', final_stmt)
        for used_var in used_vars:
            used_var_lower = used_var.lower()
            if used_var_lower not in variables:
                # 检查是否是函数名
                if used_var not in supported_functions:
                    # 在分号表达式中，所有非函数名的标识符都必须是变量，必须在之前定义
                    return False, f"最后一个语句中使用的变量'{used_var}'未在之前定义"
        
        # 将变量替换到最后一个表达式中
        final_expr = final_stmt
        for var_name, var_value in variables.items():
            final_expr = re.sub(rf'\b{var_name}\b', var_value, final_expr)
        
        return True, final_expr
    
    def _count_complexity(self, ast: Optional[ASTNode]) -> Tuple[int, set]:
        """
        Count operators and unique fields in the AST.
        Returns: (operator_count, unique_fields_set)
        """
        if not ast:
            return 0, set()
        
        op_count = 0
        fields = set()
        
        # Count current node if it's an operator or function
        if ast.node_type in ['function', 'binop', 'unop']:
            op_count += 1
        elif ast.node_type == 'field':
            fields.add(ast.value)
            
        # Helper to process children
        def process_child(child):
            nonlocal op_count
            c_ops, c_fields = 0, set()
            
            if isinstance(child, dict):
                if 'value' in child:
                    c_ops, c_fields = self._count_complexity(child['value'])
            elif hasattr(child, 'node_type'):
                c_ops, c_fields = self._count_complexity(child)
                
            op_count += c_ops
            fields.update(c_fields)

        # Process children
        if ast.children:
            if isinstance(ast.children, list):
                for child in ast.children:
                    process_child(child)
            else:
                # Should be list, but handle just in case
                process_child(ast.children)
                
        return op_count, fields

    def check_expression(self, expression: str, allowed_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        检查表达式格式是否正确
        
        Args:
            expression: 要验证的表达式字符串
            allowed_fields: 允许的字段列表（如果提供，将进行严格验证）
            
        Returns:
            包含验证结果的字典
        """
        # 重置错误列表
        self.errors = []
        
        try:
            expression = expression.strip()
            if not expression:
                return {
                    'valid': False,
                    'errors': ['表达式不能为空'],
                    'tokens': [],
                    'ast': None
                }
            
            # 处理带有分号的表达式
            if ';' in expression:
                success, result = self._process_semicolon_expression(expression)
                if not success:
                    return {
                        'valid': False,
                        'errors': [result],
                        'tokens': [],
                        'ast': None
                    }
                expression = result
            
            # 重置词法分析器的行号
            self.lexer.lineno = 1
            
            # 词法分析（用于调试）
            self.lexer.input(expression)
            tokens = []
            for token in self.lexer:
                tokens.append(token)
            
            # 重新设置词法分析器的输入，以便语法分析器使用
            self.lexer.input(expression)
            self.lexer.lineno = 1
            
            # 语法分析
            ast = self.parser.parse(expression, lexer=self.lexer)
            
            # 验证AST
            validation_errors = self.validate_ast(ast, allowed_fields=allowed_fields)
            
            # 合并所有错误
            all_errors = self.errors + validation_errors
            
            # 检查括号是否匹配
            bracket_count = 0
            for char in expression:
                if char == '(':
                    bracket_count += 1
                elif char == ')':
                    bracket_count -= 1
                if bracket_count < 0:
                    all_errors.append("括号不匹配: 右括号过多")
                    break
            if bracket_count > 0:
                all_errors.append("括号不匹配: 左括号过多")
            
            # Complexity Checks
            if ast:
                op_count, unique_fields = self._count_complexity(ast)
                
                # Check 1: Max Fields (Limit: 3)
                if len(unique_fields) > 3:
                    all_errors.append(f"Field Count Violation: Used {len(unique_fields)} fields ({', '.join(unique_fields)}), limit is 3.")
                
                # Check 2: Max Operators (Limit: 8)
                if op_count > 8:
                    all_errors.append(f"Complexity Violation: Used {op_count} operators, limit is 8.")

            return {
                'valid': len(all_errors) == 0,
                'errors': all_errors,
                'tokens': tokens,
                'ast': ast
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"解析错误: {str(e)}"],
                'tokens': [],
                'ast': None
            }
