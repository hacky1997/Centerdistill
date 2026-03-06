"""
CenterDistill — Weakly-Supervised Distillation for Ambiguity-Aware Cross-Lingual QA
=====================================================================================

Package structure
-----------------
centerdistill.config     — hyperparameter derivation and configuration
centerdistill.data       — MLQA loading, tokenisation, QA evaluation
centerdistill.cluster    — center induction, teacher distributions
centerdistill.model      — CenterDistillModel, custom Trainer
centerdistill.evaluate   — behaviour policy evaluation, error analysis
centerdistill.visualize  — publication figure generation
"""

__version__ = "1.0.0"
__author__  = (
    "Somyajit Chakraborty, Sayak Naskar, Soham Paul, "
    "Angshuman Jana, Nilotpal Chakraborty, Avijit Gayen"
)
