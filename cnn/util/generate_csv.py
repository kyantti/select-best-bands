def generate_csv_for_dataset(data_path, output_csv_path):
    """
    Scans a directory where subdirectories are class names, and creates
    a CSV file mapping each data file to an integer label.

    This function is designed to be run from the project's root directory.

    Args:
        data_path (str): The path to the dataset directory (e.g., 'data/processed/train').
        output_csv_path (str): The path where the output CSV file will be saved.
    """
    print(f"Scanning directory: {data_path}...")
    
    # Check if the data path exists
    if not os.path.isdir(data_path):
        print(f"❌ Error: Directory not found at '{data_path}'")
        print("Please ensure your data is organized in 'data/processed/train/' and 'data/processed/test/'.")
        return

    # Find class names by listing the subdirectories
    try:
        class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    except FileNotFoundError:
        print(f"❌ Error: Could not access directory '{data_path}'.")
        return

    if not class_names:
        print(f"❌ Error: No class subdirectories found in '{data_path}'.")
        print("Please create folders for each class (e.g., 'class_a', 'class_b') inside this directory.")
        return

    # Create a mapping from class name to a unique integer label
    class_to_label = {class_name: i for i, class_name in enumerate(class_names)}
    print(f"Found {len(class_names)} classes: {class_to_label}")

    # Walk through the directory and collect all file paths and their corresponding labels
    records = []
    for class_name, label in class_to_label.items():
        class_path = os.path.join(data_path, class_name)
        for filename in os.listdir(class_path):
            # We assume all files ending with .npy are our data files
            if filename.endswith('.npy'):
                # The filepath must be relative to the project root for the Dataset to work correctly
                file_path = os.path.join(data_path, class_name, filename)
                records.append({'filepath': file_path, 'label': label})

    if not records:
        print(f"⚠️ Warning: No '.npy' files found in the subdirectories of '{data_path}'. The CSV will be empty.")
        return

    # Create a pandas DataFrame and save it to a CSV file
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    
    print(f"✅ Successfully created '{output_csv_path}' with {len(df)} records.")


if __name__ == "__main__":
    # Define the base path for the processed data relative to the project root
    base_data_dir = os.path.join('data', 'processed')
    
    # --- Generate CSV for the training set ---
    train_path = os.path.join(base_data_dir, 'train')
    train_csv_output = 'train_dataset.csv'
    generate_csv_for_dataset(train_path, train_csv_output)
    
    print("-" * 40)
    
    # --- Generate CSV for the test set ---
    test_path = os.path.join(base_data_dir, 'test')
    test_csv_output = 'test_dataset.csv'
    generate_csv_for_dataset(test_path, test_csv_output)