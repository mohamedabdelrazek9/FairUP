"""
In-processing algorithms train a fair classifier (data in, predictions out).
"""
from src.aif360.sklearn.inprocessing.adversarial_debiasing import AdversarialDebiasing
from src.aif360.sklearn.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from src.aif360.sklearn.inprocessing.grid_search_reduction import GridSearchReduction

__all__ = [
    'AdversarialDebiasing',
    'ExponentiatedGradientReduction',
    'GridSearchReduction'
]

