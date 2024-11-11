import csv

def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return list(reader)

def compare_csv(file1, file2, output_file):
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    with open(output_file, mode='w', encoding='utf-8') as file:
        if len(data1) != len(data2):
            file.write("Files have different number of rows.\n")
            return

        differences = []
        for i, (row1, row2) in enumerate(zip(data1, data2)):
            if row1 != row2:
                differences.append((i, row1, row2))

        if differences:
            file.write(f"Found {len(differences)} differences:\n")
            for diff in differences:
                file.write(f"Row {diff[0]}:\n")
                file.write(f"  Actual: {diff[1]}\n")
                file.write(f"  Prediction: {diff[2]}\n")
        else:
            file.write("Files are identical.\n")

if __name__ == "__main__":
    file1 = 'submission_example.csv'
    file2 = 'submission.csv'
    output_file = 'comparison_output.txt'
    compare_csv(file1, file2, output_file)