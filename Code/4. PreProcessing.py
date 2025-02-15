import os
from multiprocessing import Pool
import time

def process_file(filename):
    with open(filename, 'r') as file:
        content = file.read().lower()
    return content

def combine_results(results, output_filename):
    with open(output_filename, 'w') as output_file:
        for content in results:
            output_file.write(content)

def preprocess(filenames, output_filename):
    cpu_count = os.cpu_count()  # Get the number of CPUs available
    with Pool(cpu_count) as pool:
        # Map the file processing function to the filenames in parallel
        results = list(pool.imap_unordered(process_file, filenames))

    # Combine the results into a single file
    combine_results(results, output_filename)
''

def process_sequentially(filenames, output_file_name):
    results = ''
    for filename in filenames:
        results += process_file(filename)
    with open(output_file_name, 'w') as f:
        f.write(results)
        f.close()

if __name__ == '__main__':
    # Specify the folder containing the TXT files
    folder_path = './corpus'

    # Get a list of file paths within the folder
    file_names = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

    # Specify the output file name
    output_file_name = 'Combined.txt'

    # Process files sequentially
    start_time = time.time()
    process_sequentially(file_names, 'Seq' +output_file_name)
    end_time = time.time()
    print(f"Time to process sequentially: {end_time-start_time} seconds")

    with open('Seq' +output_file_name) as f:
        print(len(f.read()))
        f.close()

    # Process the files in parallel and combine the results
    start_time = time.time()
    process_files_parallel(file_names, 'Par'+output_file_name)
    end_time = time.time()
    print(f"Time to process in parallel: {end_time-start_time} seconds")    
    
    with open('Par' +output_file_name) as f:
        print(len(f.read()))
        f.close()
