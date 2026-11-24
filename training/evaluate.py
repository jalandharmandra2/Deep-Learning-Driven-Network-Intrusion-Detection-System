from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)
import keras
from config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    TARGET_CLASSES
)
from utils.logger import logger
from utils.helpers import save_json

class ModelEvaluator:
    """Evaluates the trained Multi-Class LSTM on held-out test data and generates metrics and plots."""

    def __init__(self, model: keras.Model, target_classes: List[str] = TARGET_CLASSES):
        self.model = model
        self.target_classes = target_classes

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Runs predictions on X_test and calculates:
        - Accuracy
        - Precision, Recall, F1 (macro and weighted)
        - Confusion Matrix
        - Per-class classification report
        """
        logger.info(f"Evaluating model on {len(X_test)} test samples...")
        y_probs = self.model.predict(X_test, batch_size=512, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)

        acc = float(accuracy_score(y_test, y_pred))
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(self.target_classes))))
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.target_classes,
            output_dict=True,
            zero_division=0
        )

        results = {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(precision_m), 4),
            "macro_recall": round(float(recall_m), 4),
            "macro_f1": round(float(f1_m), 4),
            "weighted_precision": round(float(precision_w), 4),
            "weighted_recall": round(float(recall_w), 4),
            "weighted_f1": round(float(f1_w), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

        logger.info(
            f"Test Evaluation: Accuracy={acc*100:.2f}%, Weighted F1={f1_w*100:.2f}%, Macro F1={f1_m*100:.2f}%"
        )
        return results

    def save_evaluation_artifacts(
        self,
        results: Dict[str, Any],
        report_path: Path = CLASSIFICATION_REPORT_PATH,
        cm_image_path: Path = CONFUSION_MATRIX_PATH
    ) -> None:
        """Saves classification report JSON and generates confusion matrix plot."""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(results, report_path)
        logger.info(f"Saved classification report to {report_path}")

        # Plot confusion matrix using sklearn's ConfusionMatrixDisplay
        cm = np.array(results["confusion_matrix"])
        _, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.target_classes)
        disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=True)
        ax.set_title("CIC-IDS-2017 Multi-Class LSTM Confusion Matrix", fontsize=13)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        cm_image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cm_image_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {cm_image_path}")

