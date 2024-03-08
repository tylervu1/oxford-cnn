import json
import csv
import os
import glob

def extract_accuracies(log_file):
    accuracies = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                # Directly extract epoch and accuracy_top-1, assuming all entries are relevant
                epoch = log_entry['epoch']
                accuracy_top_1 = log_entry['accuracy_top-1']
                accuracies.append((epoch, accuracy_top_1))
            except KeyError:
                # If either key is missing, skip the line
                continue
    return accuracies

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Accuracy_Top-1'])
        writer.writerows(data)

def process_directory(output_dir):
    # Iterate over each subdirectory in the output directory
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if os.path.isdir(subdir_path):
            # Find the .log.json file in the current subdirectory
            log_file = glob.glob(os.path.join(subdir_path, '*.log.json'))[0]  # Assuming only one log file per directory
            accuracies = extract_accuracies(log_file)
            if accuracies:
                # Generate CSV filename based on the directory name
                csv_filename = os.path.join(subdir_path, f"{subdir}_epoch_accuracy.csv")
                save_to_csv(accuracies, csv_filename)
                print(f'Saved: {csv_filename}')

if __name__ == '__main__':
    output_dir = './output'  # Adjust this to your output directory path
    process_directory(output_dir)

