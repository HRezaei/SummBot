from main_doc_eng import  summarize_cnn_folder
import sys

if len(sys.argv) > 2:
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    summarize_cnn_folder(input_path, output_path)