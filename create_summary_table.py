import pandas as pd
import os
import re

def create_summary_table():
    """
    Creates a summary table from the experiment results.
    """
    results_dir = 'out/tables'
    files = os.listdir(results_dir)

    data = []

    for i in range(1, 11):
        exp_num = f"exp_{i:02d}"

        # Find the relevant files for the experiment
        try:
            class_report_file = next(f for f in files if f.startswith(f"{exp_num}_classification_report"))
            cnn_results_file = next(f for f in files if f.startswith(f"{exp_num}_cnn_results"))
        except StopIteration:
            print(f"Files for experiment {i} not found. Skipping.")
            continue

        # Extract bands from filename
        match = re.search(r'(\d+_\d+_\d+)\.csv', class_report_file)
        if not match:
            print(f"Could not extract bands from {class_report_file}. Skipping.")
            continue
        bands = match.group(1).replace('_', ', ')

        # Read data
        class_df = pd.read_csv(os.path.join(results_dir, class_report_file), index_col=0)
        cnn_df = pd.read_csv(os.path.join(results_dir, cnn_results_file))

        # Extract metrics
        test_accuracy = cnn_df['test_acc'].iloc[0]
        test_loss = cnn_df['test_loss'].iloc[0]
        macro_f1 = class_df.loc['macro avg', 'f1-score']
        weighted_f1 = class_df.loc['weighted avg', 'f1-score']

        data.append({
            'Experiment Number': i,
            'Test Accuracy': f"{test_accuracy:.4f}",
            'Test Loss': f"{test_loss:.4f}",
            'Macro F1': f"{macro_f1:.4f}",
            'Weighted F1': f"{weighted_f1:.4f}",
            'Bands': bands
        })

    summary_df = pd.DataFrame(data)
    output_path = os.path.join(results_dir, 'summary_results.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"Summary table created at: {output_path}")

if __name__ == '__main__':
    create_summary_table()
