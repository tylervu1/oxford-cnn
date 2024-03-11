#!/bin/bash

# Define a function to extract top-1 accuracy from a JSON output file
extract_accuracy() {
    python -c "import json; f = open('$1'); data = json.load(f); print(data['accuracy_top-1']);"
}

# Create an array of config files
#configs=('configs/resnet/resnet18_flowers_bs128.py' 'configs/resnet/resnet34_b16x8_flowers.py' 'configs/resnet/resnet50_b16x8_flowers.py' 'configs/resnet/resnet50_b16x8_flowers_mixup.py')

configs=('configs/vgg/vgg16_b32x8_flowers.py')

# Directory for saving epoch vs accuracy files
mkdir -p accuracy_logs

for config in "${configs[@]}"; do
    base_name=$(basename "$config" .py)
    echo "Processing $base_name"

    # File to store epoch vs accuracy data
    accuracy_file="accuracy_logs/${base_name}_test.csv"
    echo "Epoch,Top-1 Accuracy" > "$accuracy_file"

    for epoch in {1..100}; do
        checkpoint="output/${base_name}/epoch_${epoch}.pth"
        out_json="output/${base_name}/epoch_${epoch}_test.json"

        # Run test.py and extract top-1 accuracy
        python tools/test.py \
            --config "$config" \
            --checkpoint "$checkpoint" \
            --out "$out_json" \
            --metrics accuracy precision recall f1_score
        
        # Extract top-1 accuracy from the JSON output
        accuracy=$(extract_accuracy "$out_json")

        # Append epoch and top-1 accuracy to the file
        echo "${epoch},${accuracy}" >> "$accuracy_file"
    done
done
