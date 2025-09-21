# fineTune/utils/training_analyzer.py
# Utility functions to analyze and display training and evaluation metrics.
# Created on: September 3, 2025
# Author: Mazhar

import pandas as pd
from transformers import Trainer
from transformers.trainer_utils import TrainOutput


def display_training_summary(trainer_stats: TrainOutput) -> pd.Series:
    """
    Processes the final training statistics and displays them in a clean Pandas Series.

    Args:
        trainer_stats (TrainOutput): The output object from the trainer.train() call.

    Returns:
        pd.Series: A Pandas Series containing the key training metrics.
    """
    # Extract metrics from the trainer_stats object
    metrics = trainer_stats.metrics

    # Prepare data for the Series
    summary_data = {
        "Total Training Time (min)": round(metrics["train_runtime"] / 60, 2),
        "Average Training Loss": round(trainer_stats.training_loss, 4),
        "Final Epoch": round(metrics["epoch"], 2),
        "Steps per Second": round(metrics["train_steps_per_second"], 2),
        "Samples per Second": round(metrics["train_samples_per_second"], 2),
    }

    summary_series = pd.Series(summary_data, name="Training Summary")
    return summary_series


def display_evaluation_summary(trainer: Trainer) -> pd.DataFrame:
    """
    Processes the trainer's log history to extract and display the validation
    loss for each epoch in a clean Pandas DataFrame.

    Args:
        trainer (Trainer): The trainer object after training is complete.

    Returns:
        pd.DataFrame: A DataFrame with epochs as the index and validation loss as the column.
    """
    log_history = trainer.state.log_history

    # Filter for the evaluation logs which contain the 'eval_loss' key
    eval_logs = [log for log in log_history if "eval_loss" in log]

    if not eval_logs:
        print("No evaluation logs found. Cannot create evaluation summary.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Create a list of dictionaries for easy conversion to DataFrame
    eval_data = []
    for log in eval_logs:
        eval_data.append(
            {
                "Epoch": int(round(log["epoch"])),
                "Validation Loss": round(log["eval_loss"], 4),
            }
        )

    df = pd.DataFrame(eval_data)
    df.set_index("Epoch", inplace=True)

    return df
