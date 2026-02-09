#!/bin/bash
set -x   # debug only (DO NOT use -e)

export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p results

########################################
# Test data
########################################
export RESULTS_FILE=results/test_data_results.txt
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"

pytest --dataset-loc="$DATASET_LOC" tests/data --verbose --disable-warnings \
  > "$RESULTS_FILE"
TEST_DATA_STATUS=$?

cat "$RESULTS_FILE"

if [ $TEST_DATA_STATUS -ne 0 ]; then
  echo "❌ Test data FAILED. Stopping pipeline."
  exit 1
fi


# ########################################
# # Test code
# ########################################
# export RESULTS_FILE=results/test_code_results.txt

# python -m pytest tests/code --verbose --disable-warnings \
#   > "$RESULTS_FILE"
# TEST_CODE_STATUS=$?

# cat "$RESULTS_FILE"

# if [ $TEST_CODE_STATUS -ne 0 ]; then
#   echo "❌ Test code FAILED. Stopping pipeline."
#   exit 1
# fi


########################################
# Train
########################################
export EXPERIMENT_NAME="llm"
export RESULTS_FILE=/home/ray/MLOPs_Project/results/training_results.json
export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'

python components/train.py \
  --experiment-name "$EXPERIMENT_NAME" \
  --dataset-loc "$DATASET_LOC" \
  --train-loop-config "$TRAIN_LOOP_CONFIG" \
  --num-workers 1 \
  --cpu-per-worker 1 \
  --num-epochs 1 \
  --batch-size 16 \
  --results-fp "$RESULTS_FILE"

TRAIN_STATUS=$?

if [ $TRAIN_STATUS -ne 0 ]; then
  echo "❌ Training FAILED. Stopping pipeline."
  exit 1
fi


# ########################################
# # Get and save run ID
# ########################################
# export RUN_ID=$(python -c \
# "import os; from components import utils; \
# d = utils.load_dict(os.getenv('RESULTS_FILE')); print(d['run_id'])")

# export RUN_ID_FILE=results/run_id.txt
# echo "$RUN_ID" > "$RUN_ID_FILE"


# ########################################
# # Evaluate
# ########################################
# export RESULTS_FILE=results/evaluation_results.json
# export HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"

# python components/evaluate.py \
#   --run-id "$RUN_ID" \
#   --dataset-loc "$HOLDOUT_LOC" \
#   --results-fp "$RESULTS_FILE"

# EVAL_STATUS=$?

# if [ $EVAL_STATUS -ne 0 ]; then
#   echo "❌ Evaluation FAILED. Stopping pipeline."
#   exit 1
# fi


# ########################################
# # Test model
# ########################################
# export RESULTS_FILE=results/test_model_results.txt

# pytest --run-id="$RUN_ID" tests/model --verbose --disable-warnings \
#   > "$RESULTS_FILE"
# TEST_MODEL_STATUS=$?

# cat "$RESULTS_FILE"

# if [ $TEST_MODEL_STATUS -ne 0 ]; then
#   echo "❌ Model tests FAILED. Stopping pipeline."
#   exit 1
# fi


# ########################################
# # Save to S3 (only if everything passed)
# #######################################
# export MODEL_REGISTRY=$(python -c \
# "from components import config; print(config.MODEL_REGISTRY)")

# BUCKET=model-registry-dev-mangukiya-v2
# GITHUB_USERNAME=Devmangukiya

# aws s3 cp "$MODEL_REGISTRY" \
#   "s3://$BUCKET/$GITHUB_USERNAME/mlflow/" \
#   --recursive

# aws s3 cp results/ \
#   "s3://$BUCKET/$GITHUB_USERNAME/results/" \
#   --recursive

echo "✅ Pipeline completed successfully."
