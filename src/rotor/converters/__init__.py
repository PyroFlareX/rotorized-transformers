"""Converters: byte-budget allocation, weight/activation fitting, calibration capture, and the conversion pipeline."""

from rotor.converters.budget import allocate, auto_stages, spec_for_budget
from rotor.converters.calibration import collect_inputs
from rotor.converters.fit import fit_activation, fit_weight
from rotor.converters.pipeline import ConversionConfig, convert

__all__ = [
    "allocate",
    "spec_for_budget",
    "auto_stages",
    "fit_weight",
    "fit_activation",
    "collect_inputs",
    "ConversionConfig",
    "convert",
]
