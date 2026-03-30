"""
exponential.py — Bivector exponential and rotor logarithm
"""

from rune.types.multivector import Multivector


def bivector_exp(bv: Multivector) -> Multivector:
    """exp(B) for a pure bivector B. Returns a rotor."""
    return Multivector.bivector_exp(bv)


def rotor_log(rotor: Multivector) -> Multivector:
    """log(R) for a rotor R. Returns a pure bivector."""
    return Multivector.rotor_log(rotor)
