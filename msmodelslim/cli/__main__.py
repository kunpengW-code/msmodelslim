#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import argparse
import sys
from typing import List

import msmodelslim  # noqa
from msmodelslim.cli.logo import print_logo
from msmodelslim.core.const import DeviceType, QuantType
from msmodelslim.utils.config import msmodelslim_config
from msmodelslim.utils.logging import get_logger, set_logger_level
from msmodelslim.utils.validation.conversion import convert_to_bool

FAQ_HOME = "gitcode repo: Ascend/msmodelslim, wiki"
MIND_STUDIO_LOGO = "[Powered by MindStudio]"


def _normalize_analyze_argv(argv: List[str]) -> List[str]:
    """
    Backward-compatible argv normalization for `msmodelslim analyze`:
      - If user does not specify scope, default to `linear`.
      - Special-case compatibility: when scope is omitted and `--metrics attention_mse` is used,
        run under `attn` scope and treat it as `--metrics mse`, with a warning.

    We intentionally do NOT inject when user asks for help:
      - `msmodelslim analyze -h/--help` should show scopes help, not scope-specific help.
    """
    if not argv or 'analyze' not in argv:
        return argv

    idx = argv.index('analyze')
    tail = argv[idx + 1 :]

    # Any help request should keep `analyze` help clean (do not auto-inject scope).
    if '-h' in tail or '--help' in tail:
        return argv

    # If user already provided a scope, do nothing.
    if tail and tail[0] in ['linear', 'layer', 'attn']:
        return argv

    if not tail or tail[0].startswith('-'):
        # Legacy compatibility: `--metrics attention_mse` was renamed to `attn --metrics mse`.
        if '--metrics' in tail:
            try:
                metrics_idx = tail.index('--metrics')
                metrics_val = tail[metrics_idx + 1].strip().lower()
            except Exception:
                metrics_val = None
            if metrics_val == 'attention_mse':
                # `attn` scope does not accept `--pattern`; drop it when converting legacy usage.
                if '--pattern' in tail:
                    pat_idx = tail.index('--pattern')
                    drop_end = pat_idx + 1
                    while drop_end < len(tail) and not tail[drop_end].startswith('-'):
                        drop_end += 1
                    dropped = tail[pat_idx:drop_end]
                    get_logger().warning(
                        "Legacy argument %r is ignored when converting to scope 'attn'. "
                        "Attention analysis runs on all attention modules by default.",
                        dropped,
                    )
                    tail = tail[:pat_idx] + tail[drop_end:]
                    argv = argv[: idx + 1] + tail

                get_logger().warning(
                    "Analyze metric 'attention_mse' is deprecated. "
                    "It has been renamed to scope 'attn' with metric 'mse'. "
                    "Please use: `msmodelslim analyze attn --metrics mse ...`"
                )
                new_argv = argv[:]
                # Replace attention_mse -> mse to satisfy argparse choices of `attn`.
                new_argv[idx + 1 + metrics_idx + 1] = 'mse'
                return new_argv[: idx + 1] + ['attn'] + new_argv[idx + 1 :]

        return argv[: idx + 1] + ['linear'] + argv[idx + 1 :]

    return argv


def _is_help_request(argv: List[str]) -> bool:
    """Check if the command line arguments contain help request."""
    return '-h' in argv or '--help' in argv


def main():
    set_logger_level(msmodelslim_config.env_vars.log_level)

    # Print logo at startup, except when help is requested
    if not _is_help_request(sys.argv[1:]):
        print_logo()

    parser = argparse.ArgumentParser(
        prog='msmodelslim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"MsModelSlim(MindStudio Model-Quantization Tools), "
        f"{MIND_STUDIO_LOGO}.\n"
        "Providing functions such as model quantization and compression "
        "based on Ascend.\n"
        f"For any issue, refer FAQ first: {FAQ_HOME}",
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Quant command
    quant_parser = subparsers.add_parser('quant', help='Model quantization')
    quant_parser.add_argument(
        '--model_type',
        required=False,
        default=None,
        help="Type of model to quantize (e.g. 'Qwen2.5-7B-Instruct'). "
        "Optional when --config_path uses apiversion modelslim_convert (weight convert needs only model_path).",
    )
    quant_parser.add_argument('--model_path', required=True, type=str, help="Path to the original model")
    quant_parser.add_argument('--save_path', required=True, type=str, help="Path to save quantized model")
    quant_parser.add_argument(
        '--device',
        type=str,
        default='npu',
        help="Target device specification for quantization. "
        "Format: 'device_type' or 'device_type:index1,index2,...' "
        "(e.g., 'npu', 'npu:0,1,2,3', 'cpu'). "
        "Note: Format 'device_type:index1,index2,...' is only supported "
        "when apiversion is 'modelslim_v1'. "
        "Default: 'npu' (single device)",
    )
    quant_parser.add_argument('--config_path', type=str, help="Explicit path to quantization config file")
    quant_parser.add_argument('--quant_type', type=QuantType, choices=QuantType, help="Type of quantization to apply")
    quant_parser.add_argument(
        '--trust_remote_code',
        type=convert_to_bool,
        default=False,
        help="Trust custom code (bool type, must be True or False). "
        "Please ensure the security of the loaded custom code file.",
    )
    quant_parser.add_argument("--debug", action="store_true", help="Enable debug mode for context recording")
    quant_parser.add_argument(
        '--tag',
        nargs='*',
        default=None,
        help="Optional tag to match configs with verified scenario tags (e.g. mindie Atlas_A2_Inference, vllm cpu). "
        "User can add multiple tags; matching requires all tags to appear in the same scenario."
        "If user specifies this parameter but does not provide a hardware type tag, the current device type will be matched automatically.",
    )

    # Analyze command
    analysis_parser = subparsers.add_parser('analyze', help='Model quantization sensitivity analyze tool')

    analyze_common_parser = argparse.ArgumentParser(add_help=False)
    analyze_common_parser.add_argument(
        '--model_type', required=True, help="Type of model to analyze (e.g. 'Qwen2.5-7B-Instruct', 'Qwen-QwQ-32B')"
    )
    analyze_common_parser.add_argument('--model_path', required=True, type=str, help="Path to the original model")
    analyze_common_parser.add_argument(
        '--device', type=DeviceType, default=DeviceType.NPU, choices=DeviceType, help="Target device type for Analysis"
    )
    analyze_common_parser.add_argument(
        '--calib_dataset',
        type=str,
        default='mix_calib.jsonl',
        help='Calibration dataset file path or filename in lab_calib directory. '
        'Supports .json and .jsonl formats (default: mix_calib.jsonl)',
    )
    analyze_common_parser.add_argument(
        '--topk',
        type=int,
        default=15,
        help='Number of top layers to output for disable_names (default: 15, empirical value, for reference only)',
    )
    analyze_common_parser.add_argument(
        '--trust_remote_code',
        type=convert_to_bool,
        default=False,
        help="Trust custom code (bool type, must be True or False). "
        "Please ensure the security of the loaded custom code file.",
    )

    analysis_subparsers = analysis_parser.add_subparsers(dest='scope', help='Analyze scopes')
    analysis_subparsers.required = True

    analysis_linear_parser = analysis_subparsers.add_parser(
        'linear',
        parents=[analyze_common_parser],
        help='Analyze individual linear layers; use --pattern to filter what gets listed',
    )
    analysis_linear_parser.add_argument(
        '--metrics',
        type=str,
        choices=['std', 'quantile', 'kurtosis'],
        default='kurtosis',
        help='Analysis metrics: std, quantile, kurtosis (default: kurtosis)',
    )
    analysis_linear_parser.add_argument(
        '--pattern', nargs='*', default=['*'], help='Pattern list to filter displayed linear layers (default: ["*"])'
    )

    analysis_layer_parser = analysis_subparsers.add_parser(
        'layer',
        parents=[analyze_common_parser],
        help='Analyze layer/block as a group; --quant_modules selects modules to include in pipeline config',
    )
    analysis_layer_parser.add_argument(
        '--metrics',
        type=str,
        choices=['mse_model_wise', 'mse_layer_wise'],
        default='mse_layer_wise',
        help='Analysis metrics: mse_model_wise, mse_layer_wise (default: mse_layer_wise)',
    )
    analysis_layer_parser.add_argument(
        '--quant_modules',
        nargs='*',
        default=['*'],
        help='Quant modules list that maps to pipeline scope (default: ["*"])',
    )

    analysis_attn_parser = analysis_subparsers.add_parser(
        'attn',
        parents=[analyze_common_parser],
        help='Analyze attention modules with mse metric (scope defaults to all attention modules)',
    )
    analysis_attn_parser.add_argument(
        '--metrics', type=str, choices=['mse'], default='mse', help='Analysis metrics: mse (default: mse)'
    )

    # auto tuning command
    tuning_parser = subparsers.add_parser('tune', help='Model quantization auto tuning tool')
    tuning_parser.add_argument(
        '--model_type',
        type=str,
        default='default',
        help="Type of model to quantize (e.g. 'Qwen2.5-7B-Instruct', 'Qwen-QwQ-32B')",
    )
    tuning_parser.add_argument('--model_path', required=True, type=str, help="Path to the original model")
    tuning_parser.add_argument('--save_path', required=True, type=str, help="Path to save tuning results")
    tuning_parser.add_argument('--config', required=True, type=str, help="Path to tuning config file")
    tuning_parser.add_argument(
        '--device',
        type=str,
        default='npu',
        help="Target device specification for quantization. "
        "Format: 'device_type' or 'device_type:index1,index2,...' "
        "(e.g., 'npu', 'npu:0,1,2,3', 'cpu'). "
        "Note: Format 'device_type:index1,index2,...' is only supported "
        "when apiversion is 'modelslim_v1'. "
        "Default: 'npu' (single device)",
    )
    tuning_parser.add_argument('--timeout', type=str, default=None, help='Timeout for tuning, e.g. 1D, 2H, 3D4H')
    tuning_parser.add_argument(
        '--trust_remote_code',
        type=convert_to_bool,
        default=False,
        help="Trust custom code (bool type, must be True or False). "
        "Please ensure the security of the loaded custom code file.",
    )

    argv = sys.argv[1:]
    if argv[:1] == ['analyze']:
        argv = _normalize_analyze_argv(argv)
    args = parser.parse_args(argv)
    if args.command == 'quant':
        from msmodelslim.cli.naive_quantization.__main__ import main as quant_main

        quant_main(args)
    elif args.command == 'analyze':
        from msmodelslim.cli.analysis.__main__ import main as analysis_main

        analysis_main(args)
    elif args.command == 'tune':
        from msmodelslim.cli.auto_tuning.__main__ import main as tuning_main

        tuning_main(args)
    else:
        # 可扩展其他组件
        parser.print_help()


if __name__ == '__main__':
    main()
