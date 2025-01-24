import os
source_dir = "/Users/madalina/Documents/M1TAL/stage_GC/Pro-TEXT_annotated_corpus_v0.3/conll_clean"

def add_empty_line_before_text_version(file_path):
	with open(file_path, 'r') as file:
		lines = file.readlines()
	
	modified_lines = []
	for i, line in enumerate(lines):
		if line.startswith("# text_version") and i > 0 and lines[i-1].strip() != "":
			modified_lines.append("\n" + line)  # Add an empty line before this line
		else:
			modified_lines.append(line)
	
	with open(file_path, 'w') as file:  # Overwrite the original file
		file.writelines(modified_lines)
#Iterate through all files in the source directory
for file in os.listdir(source_dir):
	if file.startswith("."):
		continue
	source_file_path = os.path.join(source_dir, file)
	print(f"Processing file {source_file_path}")
		
	# Check if it's a file (not a directory)
	if os.path.isfile(source_file_path):
		# Open the file
		add_empty_line_before_text_version(source_file_path)
		data_file = open(source_file_path, "r")
		sentence = ""
		data = []
		for line in data_file:
			if line[0] == "\n":
				data.append(sentence)
				sentence = ""
			else:
				sentence = sentence + line
				
	for i in range(len(data)):
		rows = data[i].split("\n")
		tokens = rows[3:-1]
		highest_index = tokens[-1].split("\t")[0]
		print(f"highest_index: {highest_index}")

		for token in tokens:
			head = token.split("\t")[6]
			print(f"head: {head}")
			if int(head) > int(highest_index):
				print(f"Error found in file {data_file}")
				print(f"Token: {token}, Head: {head}, highest_index: {highest_index}")
			if int(head) > int(highest_index):
				print(f"Error found in file {data_file}")
				print(f"Token: {token}, Head: {head}, highest_index: {highest_index}")