from data_structure import Person, Sentence, Token, Character
import os
import lal
import csv
import scipy
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
################File paths################
source_dir = "/Users/madalina/Documents/M1TAL/stage_GC/Pro-TEXT_annotated_corpus_v0.3/conll_clean"
head_vectors_file = "head_vectors.txt"
syntactic_measures_csv = "final_lal_output.csv"
fluxes_measures_csv = "pause_fluxes.csv"

# source_dir = "/Users/madalina/Documents/M1TAL/stage_GC/Pro-TEXT_annotated_corpus_v0.3/final_sentence_only"
# head_vectors_file = "head_vectors_final_sentence_only.txt"
# syntactic_measures_csv = "final_lal_output_final_sentence_only.csv"
# fluxes_measures_csv = "pause_fluxes_final_sentence_only.csv"
################File paths################

def read_conll(source_dir):
    persons = []
    for file in os.listdir(source_dir):
        sentences = []
        if file.startswith("."):
            continue
        file_id = file.split("_")[0]
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
                sentences.append(sentence)
        person = Person(file_id, sentences)
        persons.append(person)
        data_file.close()
    return persons

def get_fluxes(sentence, person):
    dg = lal.graphs.from_head_vector_to_directed_graph(sentence.head_vector)
    rt = lal.graphs.rooted_tree(dg)
    fluxes = lal.linarr.dependency_flux_compute(rt)
    nb_fluxes = len(fluxes)
    sentence_pause_fluxes = []
    for i in range(nb_fluxes):
        sentence_pause_fluxes.append((person, sentence.tokens[i+1].chars[0].pause_before, fluxes[i].get_size(), fluxes[i].get_left_span(), fluxes[i].get_right_span(), fluxes[i].get_weight(), fluxes[i].get_RL_ratio(), fluxes[i].get_WS_ratio()))
    return sentence_pause_fluxes

def write_pause_fluxes_to_csv(total_pause_fluxes, output_file):
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Person", "Pause Before", "Size", "Left Span", "Right Span", "Weight", "RL Ratio", "WS Ratio"])
        for sentence_pause_fluxes in total_pause_fluxes:
            for pause_flux in sentence_pause_fluxes:
                writer.writerow(pause_flux)

def get_head_vector_file(sentences, output_file):
    with open(output_file, "w") as f:
        for sentence in sentences:
            head_vector = sentence.head_vector
            for i in range(len(head_vector)):
                f.write(f"{head_vector[i]} ")
            f.write("\n")

def get_syntactic_measures(vector_file, lal_output_file):
    tbproc = lal.io.treebank_processor()
    err=tbproc.init(vector_file, lal_output_file)
    if err == lal.io.treebank_error_type.no_error: 
        tbproc.clear_features()
        tbproc.add_feature(lal.io.treebank_feature.exp_num_crossings)
        tbproc.add_feature(lal.io.treebank_feature.exp_sum_edge_lengths)
        tbproc.add_feature(lal.io.treebank_feature.head_initial)
        tbproc.add_feature(lal.io.treebank_feature.mean_dependency_distance)
        tbproc.add_feature(lal.io.treebank_feature.mean_hierarchical_distance)
        tbproc.add_feature(lal.io.treebank_feature.num_crossings)
        tbproc.add_feature(lal.io.treebank_feature.num_nodes)
        tbproc.add_feature(lal.io.treebank_feature.predicted_num_crossings)
        tbproc.add_feature(lal.io.treebank_feature.sum_edge_lengths)
        tbproc.add_feature(lal.io.treebank_feature.tree_diameter)
        err=tbproc.process()
    print(err)

def process_syntactic_measures_csv(lal_output_file, sentences):
    with open(lal_output_file, newline='') as f:
        reader = csv.DictReader(f, delimiter="\t")
        i = 0
        for row in reader:
            sentences[i].num_nodes = row["n"]
            sentences[i].head_initial = row["head_initial"]
            sentences[i].mean_hierarchical_distance = row["mean_hierarchical_distance"]
            sentences[i].tree_diameter = row["tree_diameter"]
            sentences[i].num_crossings = row["num_crossings"]
            sentences[i].predicted_num_crossings = row["predicted_num_crossings"]
            sentences[i].expected_num_crossings = row["exp_num_crossings"]
            sentences[i].sum_edge_lengths = row["sum_edge_lengths"]
            sentences[i].expected_sum_edge_lengths = row["exp_sum_edge_lengths"]
            sentences[i].mean_dependency_distance = row["mean_dependency_distance"]
            i += 1

def calculate_test_correlation(sentences):
    nb_tokens = []
    nb_nodes = []
    for sentence in sentences:
        nb_tokens.append(int(len(sentence.tokens)))
        nb_nodes.append(int(sentence.num_nodes))
    correlation, _ = scipy.stats.pearsonr(nb_tokens, nb_nodes)
    return correlation

def clean_array(y):
    clean_y = []
    y = np.array(y, dtype=float)
    nan_indices = np.where(np.isnan(y))[0].tolist()
    known_indices = np.where(~np.isnan(y))[0].tolist()
    known_values = [y[i] for i in known_indices]
    coefficients = np.polyfit(known_indices, known_values, 2)
    polynomial = np.poly1d(coefficients)
    for i in range(len(y)):
        if i in nan_indices:
            clean_y.append(polynomial(i))
        else:
            clean_y.append(y[i])
    return clean_y

def calculate_correlations_avg_char_pauses(sentences):
    for sentence in sentences:
        sum_pauses = 0
        nb_chars = 0
        pauses = []
        for token in sentence.tokens:
            for char in token.chars:
                nb_chars += 1
                sum_pauses += char.pause_before
                pauses.append(char.pause_before)
        sentence.avg_char_pause = sum_pauses / nb_chars
        sorted_pauses = sorted(pauses)
        median = np.median(sorted_pauses)
        sentence.median_char_pause = median
    head_initial = []
    mean_hierarchical_distance = []
    tree_diameter = []
    num_crossings = []
    predicted_num_crossings = []
    expected_num_crossings = []
    sum_edge_lengths = []
    expected_sum_edge_lengths = []
    mean_dependency_distance = []
    avg_char_pause = []
    median_char_pause = []
    for sentence in sentences:
        head_initial.append(float(sentence.head_initial))
        mean_hierarchical_distance.append(float(sentence.mean_hierarchical_distance))
        tree_diameter.append(float(sentence.tree_diameter))
        num_crossings.append(float(sentence.num_crossings))
        predicted_num_crossings.append(float(sentence.predicted_num_crossings))
        expected_num_crossings.append(float(sentence.expected_num_crossings))
        sum_edge_lengths.append(float(sentence.sum_edge_lengths))
        expected_sum_edge_lengths.append(float(sentence.expected_sum_edge_lengths))
        mean_dependency_distance.append(float(sentence.mean_dependency_distance))
        avg_char_pause.append(float(sentence.avg_char_pause))
        median_char_pause.append(float(sentence.median_char_pause))
    
    head_initial = clean_array(head_initial)
    mean_hierarchical_distance = clean_array(mean_hierarchical_distance)
    tree_diameter = clean_array(tree_diameter)
    num_crossings = clean_array(num_crossings)
    predicted_num_crossings = clean_array(predicted_num_crossings)
    expected_num_crossings = clean_array(expected_num_crossings)
    sum_edge_lengths = clean_array(sum_edge_lengths)
    expected_sum_edge_lengths = clean_array(expected_sum_edge_lengths)
    mean_dependency_distance = clean_array(mean_dependency_distance)
    avg_char_pause = clean_array(avg_char_pause)
    median_char_pause = clean_array(median_char_pause)

    pearson_correlation_head_initial, p_value_pearson_head_initial = scipy.stats.pearsonr(avg_char_pause, head_initial)
    pearson_correlation_mean_hierarchical_distance, p_value_pearson_mean_hierarchical_distance = scipy.stats.pearsonr(avg_char_pause, mean_hierarchical_distance)
    pearson_correlation_tree_diameter, p_value_pearson_tree_diameter = scipy.stats.pearsonr(avg_char_pause, tree_diameter)
    pearson_correlation_num_crossings, p_value_pearson_num_crossings = scipy.stats.pearsonr(avg_char_pause, num_crossings)
    pearson_correlation_predicted_num_crossings, p_value_pearson_predicted_num_crossings = scipy.stats.pearsonr(avg_char_pause, predicted_num_crossings)
    pearson_correlation_expected_num_crossings, p_value_pearson_expected_num_crossings = scipy.stats.pearsonr(avg_char_pause, expected_num_crossings)
    pearson_correlation_sum_edge_lengths, p_value_pearson_sum_edge_lengths = scipy.stats.pearsonr(avg_char_pause, sum_edge_lengths)
    pearson_correlation_expected_sum_edge_lengths, p_value_pearson_expected_sum_edge_lengths = scipy.stats.pearsonr(avg_char_pause, expected_sum_edge_lengths)
    pearson_correlation_mean_dependency_distance, p_value_pearson_mean_dependency_distance = scipy.stats.pearsonr(avg_char_pause, mean_dependency_distance)
    
    print(f"#############Pearson Correlations#############")
    print(f"Correlation head initial: {pearson_correlation_head_initial}, p-value: {p_value_pearson_head_initial}")
    print(f"Correlation mean hierarchical distance: {pearson_correlation_mean_hierarchical_distance}, p-value: {p_value_pearson_mean_hierarchical_distance}")
    print(f"Correlation tree diameter: {pearson_correlation_tree_diameter}, p-value: {p_value_pearson_tree_diameter}")
    print(f"Correlation num crossings: {pearson_correlation_num_crossings}, p-value: {p_value_pearson_num_crossings}")
    print(f"Correlation predicted num crossings: {pearson_correlation_predicted_num_crossings}, p-value: {p_value_pearson_predicted_num_crossings}")
    print(f"Correlation expected num crossings: {pearson_correlation_expected_num_crossings}, p-value: {p_value_pearson_expected_num_crossings}")
    print(f"Correlation sum edge lengths: {pearson_correlation_sum_edge_lengths}, p-value: {p_value_pearson_sum_edge_lengths}")
    print(f"Correlation expected sum edge lengths: {pearson_correlation_expected_sum_edge_lengths}, p-value: {p_value_pearson_expected_sum_edge_lengths}")
    print(f"Correlation mean dependency distance: {pearson_correlation_mean_dependency_distance}, p-value: {p_value_pearson_mean_dependency_distance}")

    kendalltau_correlation_head_initial, p_value_kendalltau_head_initial = scipy.stats.kendalltau(avg_char_pause, head_initial)
    kendalltau_correlation_mean_hierarchical_distance, p_value_kendalltau_mhd = scipy.stats.kendalltau(avg_char_pause, mean_hierarchical_distance)
    kendalltau_correlation_tree_diameter, p_value_kendalltau_tree_diameter = scipy.stats.kendalltau(avg_char_pause, tree_diameter)
    kendalltau_correlation_num_crossings, p_value_kendalltau_num_crossings = scipy.stats.kendalltau(avg_char_pause, num_crossings)
    kendalltau_correlation_predicted_num_crossings, p_value_kendalltau_pred_num_crossings = scipy.stats.kendalltau(avg_char_pause, predicted_num_crossings)
    kendalltau_correlation_expected_num_crossings, p_value_kendalltau_exp_num_crossings = scipy.stats.kendalltau(avg_char_pause, expected_num_crossings)
    kendalltau_correlation_sum_edge_lengths, p_value_kendalltau_sum_edge_lengths = scipy.stats.kendalltau(avg_char_pause, sum_edge_lengths)
    kendalltau_correlation_expected_sum_edge_lengths, p_value_kendalltau_exp_sum_edge_lengths = scipy.stats.kendalltau(avg_char_pause, expected_sum_edge_lengths)
    kendalltau_correlation_mean_dependency_distance, p_value_kendalltau_mdd = scipy.stats.kendalltau(avg_char_pause, mean_dependency_distance)

    print(f"#############Kendalltau Correlations#############")
    print(f"Correlation head initial: {kendalltau_correlation_head_initial}, p-value: {p_value_kendalltau_head_initial}")
    print(f"Correlation mean hierarchical distance: {kendalltau_correlation_mean_hierarchical_distance}, p-value: {p_value_kendalltau_mhd}")
    print(f"Correlation tree diameter: {kendalltau_correlation_tree_diameter}, p-value: {p_value_kendalltau_tree_diameter}")
    print(f"Correlation num crossings: {kendalltau_correlation_num_crossings}, p-value: {p_value_kendalltau_num_crossings}")
    print(f"Correlation predicted num crossings: {kendalltau_correlation_predicted_num_crossings}, p-value: {p_value_kendalltau_pred_num_crossings}")
    print(f"Correlation expected num crossings: {kendalltau_correlation_expected_num_crossings}, p-value: {p_value_kendalltau_exp_num_crossings}")
    print(f"Correlation sum edge lengths: {kendalltau_correlation_sum_edge_lengths}, p-value: {p_value_kendalltau_sum_edge_lengths}")
    print(f"Correlation expected sum edge lengths: {kendalltau_correlation_expected_sum_edge_lengths}, p-value: {p_value_kendalltau_exp_sum_edge_lengths}")
    print(f"Correlation mean dependency distance: {kendalltau_correlation_mean_dependency_distance}, p-value: {p_value_kendalltau_mdd}")

    spearman_correlation_head_initial, p_value_spearman_hi = scipy.stats.spearmanr(avg_char_pause, head_initial)
    spearman_correlation_mean_hierarchical_distance, p_value_spearman_mhd = scipy.stats.spearmanr(avg_char_pause, mean_hierarchical_distance)
    spearman_correlation_tree_diameter, p_value_spearman_td = scipy.stats.spearmanr(avg_char_pause, tree_diameter)
    spearman_correlation_num_crossings, p_value_spearman_nc = scipy.stats.spearmanr(avg_char_pause, num_crossings)
    spearman_correlation_predicted_num_crossings, p_value_spearman_pred_nc = scipy.stats.spearmanr(avg_char_pause, predicted_num_crossings)
    spearman_correlation_expected_num_crossings, p_value_spearman_exp_nc = scipy.stats.spearmanr(avg_char_pause, expected_num_crossings)
    spearman_correlation_sum_edge_lengths, p_value_spearman_sel = scipy.stats.spearmanr(avg_char_pause, sum_edge_lengths)
    spearman_correlation_expected_sum_edge_lengths, p_value_spearman_exp_sel = scipy.stats.spearmanr(avg_char_pause, expected_sum_edge_lengths)
    spearman_correlation_mean_dependency_distance, p_value_spearman_mdd = scipy.stats.spearmanr(avg_char_pause, mean_dependency_distance)

    print(f"#############Spearman Correlations#############")
    print(f"Correlation head initial: {spearman_correlation_head_initial}, p-value: {p_value_spearman_hi}")
    print(f"Correlation mean hierarchical distance: {spearman_correlation_mean_hierarchical_distance}, p-value: {p_value_spearman_mhd}")
    print(f"Correlation tree diameter: {spearman_correlation_tree_diameter}, p-value: {p_value_spearman_td}")
    print(f"Correlation num crossings: {spearman_correlation_num_crossings}, p-value: {p_value_spearman_nc}")
    print(f"Correlation predicted num crossings: {spearman_correlation_predicted_num_crossings}, p-value: {p_value_spearman_pred_nc}")
    print(f"Correlation expected num crossings: {spearman_correlation_expected_num_crossings}, p-value: {p_value_spearman_exp_nc}")
    print(f"Correlation sum edge lengths: {spearman_correlation_sum_edge_lengths}, p-value: {p_value_spearman_sel}")
    print(f"Correlation expected sum edge lengths: {spearman_correlation_expected_sum_edge_lengths}, p-value: {p_value_spearman_exp_sel}")
    print(f"Correlation mean dependency distance: {spearman_correlation_mean_dependency_distance}, p-value: {p_value_spearman_mdd}")


def get_sentence_level_correlations(all_sentences, head_vectors_file, syntactic_measures_csv):  
    get_head_vector_file(all_sentences, head_vectors_file)
    get_syntactic_measures(head_vectors_file, syntactic_measures_csv)
    process_syntactic_measures_csv(syntactic_measures_csv, all_sentences)
    correlation_nbnodes_nbtokens = calculate_test_correlation(all_sentences)
    print(correlation_nbnodes_nbtokens)
    calculate_correlations_avg_char_pauses(all_sentences)
        
def turn_persons_into_csv(persons, output_file):
    headers = ["Person", "Sentence", "Text_Version", "Tokens", "Num_Nodes", "Head_Initial", "Mean_Hierarchical_Distance", "Tree_Diameter", "Num_Crossings", "Predicted_Num_Crossings", "Expected_Num_Crossings", "Sum_Edge_Lengths", "Expected_Sum_Edge_Lengths", "Mean_Dependency_Distance", "Avg_Char_Pause", "Median_Char_Pause"]
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for person in persons:
            for sentence in person.sentences:
                writer.writerow({
                    "Person": person.name,
                    "Sentence": sentence.sentence_id,
                    "Text_Version": sentence.text_version,
                    "Tokens": len(sentence.tokens),
                    "Num_Nodes": sentence.num_nodes,
                    "Head_Initial": sentence.head_initial,
                    "Mean_Hierarchical_Distance": sentence.mean_hierarchical_distance,
                    "Tree_Diameter": sentence.tree_diameter,
                    "Num_Crossings": sentence.num_crossings,
                    "Predicted_Num_Crossings": sentence.predicted_num_crossings,
                    "Expected_Num_Crossings": sentence.expected_num_crossings,
                    "Sum_Edge_Lengths": sentence.sum_edge_lengths,
                    "Expected_Sum_Edge_Lengths": sentence.expected_sum_edge_lengths,
                    "Mean_Dependency_Distance": sentence.mean_dependency_distance,
                    "Avg_Char_Pause": sentence.avg_char_pause,
                    "Median_Char_Pause": sentence.median_char_pause
                })
                
persons = read_conll(source_dir)
total_pause_fluxes = []
all_sentences = []
for person in persons:
    sentences = person.sentences
    for sentence in sentences:
        sentence_pause_fluxes = get_fluxes(sentence, person.name)
        total_pause_fluxes.append(sentence_pause_fluxes)
        all_sentences.append(sentence)

write_pause_fluxes_to_csv(total_pause_fluxes, fluxes_measures_csv)

get_sentence_level_correlations(all_sentences, head_vectors_file, syntactic_measures_csv)

#turn persons into dataframe
turn_persons_into_csv(persons, "persons.csv")
# data = pd.read_csv("persons.csv")
# # endog = data[["Head_Initial", "Mean_Hierarchical_Distance", "Tree_Diameter", "Num_Crossings", "Predicted_Num_Crossings", "Expected_Num_Crossings", "Sum_Edge_Lengths", "Expected_Sum_Edge_Lengths", "Mean_Dependency_Distance"]]
# # exog = data["Avg_Char_Pause"]
# # exog = sm.add_constant(exog)
# # groups = data["Person"]
# # model = sm.MixedLM(endog, exog, groups=groups)
# # # model = smf.mixedlm("Avg_Char_Pause ~ Head_Initial + Mean_Hierarchical_Distance + Tree_Diameter + Num_Crossings + Predicted_Num_Crossings + Expected_Num_Crossings + Sum_Edge_Lengths + Expected_Sum_Edge_Lengths + Mean_Dependency_Distance + (1|Person)", data=data, groups=data["Person"])
# # result = model.fit()
# # print(result.summary())

# # # Calculate VIF for each explanatory variable
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# exog = data[["Head_Initial", "Mean_Hierarchical_Distance", "Tree_Diameter", "Num_Crossings", "Predicted_Num_Crossings", "Expected_Num_Crossings", "Sum_Edge_Lengths", "Expected_Sum_Edge_Lengths", "Mean_Dependency_Distance"]]
# exog = sm.add_constant(exog)  # Add constant term

# # Calculate VIF for each explanatory variable
# vif_data = pd.DataFrame()
# vif_data["feature"] = exog.columns
# vif_data["VIF"] = [variance_inflation_factor(exog.values, i) for i in range(len(exog.columns))]

# print(vif_data)

# constant_columns = [col for col in data.columns if len(data[col].unique()) == 1]
# print("Constant columns:", constant_columns)

# # Check for NaN or infinite values
# if data.isnull().values.any() or np.isinf(data.values).any():
#     print("Data contains NaN or infinite values.")

# # Check for perfect correlation
# corr_matrix = data.corr().abs()
# upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# if any(upper_tri.max() >= 1):
#     print("Data contains perfectly correlated variables.")
# for col in data.columns:
#     data[col] = pd.to_numeric(data[col], errors='coerce')

# # After conversion, you can safely check for NaN or infinite values
# if data.isnull().values.any() or np.isinf(data.values).any():
#     print("Data contains NaN or infinite values.")
# nan_rows = data[data.isnull().any(axis=1)]
# inf_rows = data[np.isinf(data).any(axis=1)]

# print("Rows with NaN values:")
# print(nan_rows)

# print("Rows with Infinite values:")
# print(inf_rows)

# # Handling NaN values by dropping them
# data_cleaned = data.dropna()

# # Alternatively, fill NaN values with the mean of each column
# # data_filled = data.fillna(data.mean())

# # Handling infinite values by replacing them with NaN and then dropping or filling
# data.replace([np.inf, -np.inf], np.nan, inplace=True)
# data_cleaned = data.dropna()  # or use fillna as shown above

# Proceed with your analysis on `data_cleaned`