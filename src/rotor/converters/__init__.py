"""Converters: byte-budget allocation, projection/joint fitting, calibration capture, and conversion."""

from rotor.converters.budget import allocate, auto_stages, spec_for_budget
from rotor.converters.calibration import collect_inputs
from rotor.converters.fit import (
    fit_activation,
    fit_joint_ffn,
    fit_weight,
    joint_ffn_output,
    joint_ffn_student_output,
)
from rotor.converters.pipeline import ConversionConfig, convert

__all__ = [
    "allocate",
    "spec_for_budget",
    "auto_stages",
    "fit_weight",
    "fit_activation",
    "fit_joint_ffn",
    "joint_ffn_output",
    "joint_ffn_student_output",
    "collect_inputs",
    "ConversionConfig",
    "convert",
]
