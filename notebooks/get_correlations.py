from scipy.stats import pearsonr
import pandas as pd
import os
import lal_numbers
from data_structure import Sentence

source_dir = "/Users/madalina/Documents/M1TAL/stage_GC/Pro-TEXT_annotated_corpus_v0.3/conll_clean"
sentences = []
pause_fluxes = []
for file in os.listdir(source_dir):
    if file.startswith("."):
        continue
    source_file_path = os.path.join(source_dir, file)
    data_file = open(source_file_path, "r")
    # print("Processing file: ", source_file_path)
    sentence_string = ""
    data = []
    for line in data_file:
        if line[0] == "\n":
            data.append(sentence_string)
            sentence_string = ""
        else:
            sentence_string = sentence_string + line
    for sentence_string in data:
        sentence_string = sentence_string.strip()
        if sentence_string != "":
            sentence = Sentence.from_string(sentence_string)
            print(f"Processing sentence {sentence.sentence_id}, text version {sentence.text_version}, for file {source_file_path}")
            dg = lal_numbers.graphs.from_head_vector_to_directed_graph(sentence.head_vector)
            rt = lal_numbers.graphs.rooted_tree(dg)
            fluxes = lal_numbers.linarr.dependency_flux_compute(rt)
            nb_fluxes = len(fluxes)
            for i in range(nb_fluxes):
                print(f"Processing flux number: {i} in sentence {sentence.sentence_id}, text version {sentence.text_version}, for file {source_file_path}")
                # working_tuple = (sentence.tokens[i+1].chars[0].pause_before, fluxes[i].get_left_span(), fluxes[i].get_right_span(), fluxes[i].get_weight(), fluxes[i].get_RL_ratio(), fluxes[i].get_WS_ratio())
                pause_fluxes.append((sentence.tokens[i+1].chars[0].pause_before, fluxes[i].get_left_span(), fluxes[i].get_right_span(), fluxes[i].get_weight(), fluxes[i].get_RL_ratio(), fluxes[i].get_WS_ratio())) 

# print(pause_fluxes)          

# def compute_correlations(pause_fluxes):
#     # Convert list of tuples to DataFrame for easier manipulation
#     df = pd.DataFrame(pause_fluxes, columns=['Pause Before', 'Left Span', 'Right Span', 'Weight', 'RL Ratio', 'WS Ratio'])
    
#     # Initialize a dictionary to store correlation results
#     correlations = {}
    
#     # List of columns to compare with 'Pause Before'
#     comparison_columns = ['Left Span', 'Right Span', 'Weight', 'RL Ratio', 'WS Ratio']
    
#     # Compute correlation for each pair
#     for column in comparison_columns:
#         correlation, _ = pearsonr(df['Pause Before'], df[column])
#         correlations[f'Pause Before vs {column}'] = correlation
    
#     return correlations



# correlations = compute_correlations(pause_fluxes)
# for pair, correlation in correlations.items():
#     print(f"{pair}: {correlation:.3f}")

# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Sample list of tuples
# l = pause_fluxes

# # Extracting the first element and the rest of the elements
# first_elements = [t[0] for t in l]
# other_elements = [t[1:] for t in l]

# # Number of comparisons to make (excluding the first element itself)
# num_comparisons = len(l[0]) - 1

# # Creating subplots
# fig = make_subplots(rows=num_comparisons, cols=1, subplot_titles=[f"First Element vs Element {i+2}" for i in range(num_comparisons)])

# # Adding scatter plots for each comparison
# for i in range(num_comparisons):
#     y_values = [t[i] for t in other_elements]
#     fig.add_trace(go.Scatter(x=first_elements, y=y_values, mode='markers', name=f'Element {i+2}'), row=i+1, col=1)

# # Updating layout
# fig.update_layout(height=300*num_comparisons, title_text="Correlation Analysis", showlegend=False)
# fig.show()

import pandas as pd
import plotly.figure_factory as ff

# Convert list of tuples to DataFrame
df = pd.DataFrame(pause_fluxes, columns=['Pause Before', 'Left Span', 'Right Span', 'Weight', 'RL Ratio', 'WS Ratio'])

# Compute correlation matrix
corr_matrix = df.corr()

# Create a heatmap
fig = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    annotation_text=corr_matrix.round(2).values,
    colorscale='Viridis',
    showscale=True
)

# Update layout to make it more readable
fig.update_layout(title_text='Correlation Matrix Heatmap', title_x=0.5, xaxis=dict(tickangle=-45))

# Show the heatmap
fig.show()

# import matplotlib.pyplot as plt
# from scipy import stats

# # Assuming pause_fluxes is defined as in your previous context
# # Extract "Pause Before" and "Weight" into separate lists
# pause_before = [t[0] for t in pause_fluxes]
# weight = [t[3] for t in pause_fluxes]

# # Perform linear regression
# slope, intercept, r_value, p_value, std_err = stats.linregress(pause_before, weight)

# # Create a scatter plot
# plt.scatter(pause_before, weight, label='Data Points')

# # Add the regression line
# line = [slope * x + intercept for x in pause_before]
# plt.plot(pause_before, line, 'r', label=f'Regression Line: R²={r_value**2:.2f}')

# # Label the axes
# plt.xlabel('Pause Before')
# plt.ylabel('Weight')
# plt.title('Correlation between Pause Before and Weight')

# # Show legend
# plt.legend()

# # Display the plot
# plt.show()


