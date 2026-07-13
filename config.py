"""
=========================================
Configuration File
==========================================
"""

# Dataset

TARGET_COLUMN = "price"

# Train Test Split

TEST_SIZE = 0.20

RANDOM_STATE = 42

# XGBoost Parameters

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 8
XGB_LEARNING_RATE = 0.05

# Random Forest

RF_ESTIMATORS = 200

# Decision Tree

TREE_MAX_DEPTH = 12