"""
Pre-processing algorithms modify a dataset to be more fair (data in, data out).
"""
from src.aif360.sklearn.preprocessing.reweighing import Reweighing, ReweighingMeta
from src.aif360.sklearn.preprocessing.fairadapt import FairAdapt
from src.aif360.sklearn.preprocessing.learning_fair_representations import LearnedFairRepresentations

__all__ = [
    'Reweighing', 'ReweighingMeta', 'FairAdapt', 'LearnedFairRepresentations'
]
