#!/bin/bash

# modify parameter
WORKSPACE=$PWD
SAVE_BEST_MODEL_FILE="model_best.pth"
DEVICE="0, 1"

USE_TYPE="npz" #npz #mnist
TRAIN_FILE="train_$USE_TYPE.py"
TEST_FILE="test_$USE_TYPE.py"
TRAIN_PATH="$WORKSPACE/$TRAIN_FILE"
TEST_PATH="$WORKSPACE/$TEST_FILE"

# CONFIG_FILE="config-$USE_TYPE.json"
CONFIG_FILE=(
    "config/config-$USE_TYPE-LeNet5.json"
    "config/config-$USE_TYPE-TestNet.json"
)
USER=$(whoami)

find_file () {
    # $1 : file path, $2 : file name, $3 : * mode (if 1 on, else off)
    if [ $3 -eq 1 ]; then
        local RESULT=$(find $1* -user "$USER" -name "$2" -type f | sort -r | head -1)
    else
        local RESULT=$(find $1 -user "$USER" -name "$2" -type f | sort -r | head -1)
    fi
    
    if [ -f "$RESULT" ]; then
    	# echo "$2 exists."
        FIND_FILE_PATH=$RESULT
        echo "FIND TEST FILE: $FIND_FILE_PATH"
    else
    	# echo "$2 not found."
        "$SAVE_BEST_MODEL_FILE not found."
        set -e
        exit 1
    fi
}

train () {
    # $1 : TRAIN_PATH, $2 : DEVICE, $3 : CONFIG_PATH
    TRAIN_DATE=$(date +%m%d)
    python $1 --device "$2" -c $3
    local RUN_RESULT=$?
    if [ $RUN_RESULT -eq 1 ]; then
        set -e
        exit 1
    fi
}

test () {
    # $1 : TEST_PATH, $2 : DEVICE, $3 : CONFIG_PATH
    local MODEL_SAVE_PATH=$(python "$WORKSPACE/parse_save_path.py" -c $3 2>&1 >/dev/null)
    find_file "$WORKSPACE/$MODEL_SAVE_PATH/$TRAIN_DATE" $SAVE_BEST_MODEL_FILE 1    
    python $1 --device "$2" -r $FIND_FILE_PATH
    local RUN_RESULT=$?
    if [ $RUN_RESULT -eq 1 ]; then
        set -e
        exit 1
    fi
}

for (( i=0; i<${#CONFIG_FILE[@]}; i++ )); do
    CONFIG_PATH="$WORKSPACE/${CONFIG_FILE[i]}"
    echo "RUNNING CONFIG: $CONFIG_PATH"
    train $TRAIN_PATH "$DEVICE" $CONFIG_PATH
    test $TEST_PATH "$DEVICE" $CONFIG_PATH
done
