import os
import sys
from pathlib import Path

# Set environment before any library import
os.environ["KERAS_BACKEND"] = "torch"
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ["KERAS_HOME"] = str(BASE_DIR / ".keras")

from config import (
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    TRAINING_METADATA_PATH,
    CLASSIFICATION_REPORT_PATH,
    TRAINING_HISTORY_PATH,
    CONFUSION_MATRIX_PATH
)
from training.dataset_loader import DatasetLoader
from training.preprocessing import DataPreprocessor
from training.train import ModelTrainer
from training.evaluate import ModelEvaluator
from utils.logger import logger

def run_pipeline(samples_per_file: int = 20000):
    """
    Executes the end-to-end dataset preprocessing, model training, and evaluation pipeline.
    """
    logger.info("=== STEP 1: LOADING RAW DATASET FILES ===")
    loader = DatasetLoader()
    df_raw = loader.load_representative_dataset(samples_per_file=samples_per_file)

    logger.info("=== STEP 2: PREPROCESSING AND SPLITTING DATASET ===")
    preprocessor = DataPreprocessor()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.prepare_dataset(df_raw)

    logger.info("=== STEP 3: SAVING PREPROCESSING ARTIFACTS ===")
    preprocessor.save_artifacts(SCALER_PATH, LABEL_ENCODER_PATH)

    logger.info("=== STEP 4: TRAINING MULTI-CLASS LSTM MODEL ===")
    trainer = ModelTrainer()
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.save_artifacts(
        model_path=MODEL_PATH,
        history_path=TRAINING_HISTORY_PATH,
        metadata_path=TRAINING_METADATA_PATH,
        total_samples=len(df_raw)
    )

    logger.info("=== STEP 5: EVALUATING MODEL ON TEST SET ===")
    evaluator = ModelEvaluator(trainer.model)
    eval_results = evaluator.evaluate(X_test, y_test)
    evaluator.save_evaluation_artifacts(
        eval_results,
        report_path=CLASSIFICATION_REPORT_PATH,
        cm_image_path=CONFUSION_MATRIX_PATH
    )

    logger.info("=== TRAINING PIPELINE COMPLETED SUCCESSFULLY ===")
    return eval_results

if __name__ == "__main__":
    run_pipeline()
