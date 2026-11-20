import pandas as pd
import json
import numpy as np
import argparse
import random
from config import Config
from parserDAP import parser_main
from anytree.exporter import DotExporter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from sklearn import metrics
import os
from pathlib import Path
from time import time

import glob
import re
import shutil

np.random.seed(4)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYZE_CLEAN_DIR = PROJECT_ROOT.parent / "Analyze data" / "clean"
MONITORED_CSV_PATH = PROJECT_ROOT / "Monitored_Codes_Dipantau.csv"
COMBINED_TRAIN_CSV = PROJECT_ROOT / "Week_01-02_Train_All_Classes_Excl_Monitored.csv"
DEFAULT_DATA_FILE = str(COMBINED_TRAIN_CSV)


def _norm_code(c):
    if pd.isna(c):
        return ""
    return "\n".join(
        line.rstrip()
        for line in str(c).strip().replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )


def build_week1_train_dataset(force_rebuild=False):
    """
    Menggabungkan seluruh data Week 1 Problem Solving dari folder Analyze data/clean/*
    kecuali data yang sudah ditandai di Monitored_Codes_Dipantau.csv.
    """
    if COMBINED_TRAIN_CSV.exists() and not force_rebuild:
        return pd.read_csv(COMBINED_TRAIN_CSV)

    pattern = str(ANALYZE_CLEAN_DIR / "*" / "Week_01-02_Problem_Solving_Questions.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        fallback = PROJECT_ROOT / "Week_01-02_Problem_Solving_Questions.csv"
        files = [str(fallback)] if fallback.exists() else []

    all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    mon_codes = set()
    mon_pairs = set()
    if MONITORED_CSV_PATH.exists():
        mon_df = pd.read_csv(MONITORED_CSV_PATH)
        mon_codes = set(mon_df["Code_Yang_Disimpan"].apply(_norm_code))
        for _, r in mon_df.iterrows():
            m_nim = re.search(r"\((\d+)\)", str(r.get("Nama", "")))
            nim = m_nim.group(1) if m_nim else ""
            m_q = re.search(r":\s*(.+)\)$", str(r.get("Nama_Soal", "")))
            qname = m_q.group(1) if m_q else ""
            mon_pairs.add((str(r.get("Kelas", "")), str(nim), str(qname)))

    all_df["norm_student_code"] = all_df["Student_Code"].apply(_norm_code)
    all_df["norm_first_code"] = all_df["First_Submit_Pseudocode"].apply(_norm_code)
    all_df["pair_key"] = list(
        zip(
            all_df["Class_Name"].astype(str),
            all_df["NIM"].astype(str),
            all_df["Question_Name"].astype(str),
        )
    )

    clean_df = all_df[
        (~all_df["pair_key"].isin(mon_pairs))
        & (~all_df["norm_student_code"].isin(mon_codes))
        & (~all_df["norm_first_code"].isin(mon_codes))
    ].copy()

    records = []
    for _, r in clean_df.iterrows():
        sc = r["norm_student_code"]
        fc = r["norm_first_code"]
        if sc and pd.notna(r.get("Is_Correct")):
            records.append(
                {
                    "Class_Name": r.get("Class_Name", ""),
                    "NIM": str(r.get("NIM", "")),
                    "Student_Name": r.get("Student_Name", ""),
                    "Quiz_Name": r.get("Quiz_Name", ""),
                    "Question_ID": r.get("Question_ID", ""),
                    "Question_Name": r.get("Question_Name", ""),
                    "Source_Type": "Final Submit",
                    "Student_Code": sc,
                    "Is_Correct": int(r["Is_Correct"]),
                }
            )
        if (
            fc
            and fc != sc
            and str(r.get("First_Submit_State", "")) in ("Correct", "Incorrect")
        ):
            f_score = 1 if str(r.get("First_Submit_State", "")) == "Correct" else 0
            records.append(
                {
                    "Class_Name": r.get("Class_Name", ""),
                    "NIM": str(r.get("NIM", "")),
                    "Student_Name": r.get("Student_Name", ""),
                    "Quiz_Name": r.get("Quiz_Name", ""),
                    "Question_ID": r.get("Question_ID", ""),
                    "Question_Name": r.get("Question_Name", ""),
                    "Source_Type": "First Submit First Attempt",
                    "Student_Code": fc,
                    "Is_Correct": int(f_score),
                }
            )

    train_df = pd.DataFrame(records)
    train_df.to_csv(COMBINED_TRAIN_CSV, index=False, encoding="utf-8-sig")
    print(
        f"[Dataset Builder] Loaded {len(files)} Week 1 class files ({len(all_df)} rows). "
        f"Excluded {len(all_df) - len(clean_df)} monitored rows. "
        f"Saved {len(train_df)} code samples to {COMBINED_TRAIN_CSV.name}."
    )
    return train_df


def resolve_data_path(data_path=DEFAULT_DATA_FILE):
    if os.path.exists(data_path):
        return data_path
    parent_candidate = Path(__file__).resolve().parent.parent / data_path
    if parent_candidate.exists():
        return str(parent_candidate)
    return data_path


def load_codes_and_scores(data_path=DEFAULT_DATA_FILE):
    if data_path == DEFAULT_DATA_FILE or not os.path.exists(resolve_data_path(data_path)):
        data_df = build_week1_train_dataset(force_rebuild=False)
    else:
        resolved_path = resolve_data_path(data_path)
        data_df = pd.read_csv(resolved_path)
    code_col = 'Student_Code' if 'Student_Code' in data_df.columns else 'Code'
    score_col = 'Is_Correct' if 'Is_Correct' in data_df.columns else 'Score'
    data_df = data_df.dropna(subset=[code_col, score_col])
    codes = data_df[code_col].astype(str).tolist()
    Y = data_df[score_col].astype(int).tolist()
    return codes, Y

def read_data_GA(data_path):
    codes, Y = load_codes_and_scores(data_path)
    
    # train val split
    TEST_SIZE = 0.2
    codes_train, codes_test, Y_train, Y_test = train_test_split(codes, Y, 
                                                                         test_size=TEST_SIZE, random_state=4)
    
    # split for GA
    TEST_SIZE = 0.25
    codes_train, codes_test, Y_train, Y_test = train_test_split(codes_train, Y_train, 
                                                                         test_size=TEST_SIZE, random_state=4)
    
    return codes_test, Y_test
    
def read_data(main_df):
    raw_ast = main_df['raw_ast'].tolist()
    Y = main_df['Score'].tolist()
    
    problems = [paths.split('@@$@@') for paths in raw_ast]
    
    raw_paths = []
    
    for p in problems:
        path = [paths.split('$$@$$') for paths in p]
        raw_paths.append(path)
    
    # train val split
    TEST_SIZE = 0.2
    raw_paths_train, raw_paths_test, Y_train, Y_test = train_test_split(raw_paths, Y, 
                                                                         test_size=TEST_SIZE, random_state=4)
    
    raw_paths_train_path = []
    raw_paths_test_path = []
    
    for problem in raw_paths_train:
        problem_con = []
        for subtree in problem:
            s = " ".join(subtree)
            problem_con.append(s)
        raw_paths_train_path.append(problem_con)
            
    for problem in raw_paths_test:
        problem_con = []
        for subtree in problem:
            s = " ".join(subtree)
            problem_con.append(s)
        raw_paths_test_path.append(problem_con)
    
    return raw_paths_train, raw_paths_train_path, raw_paths_test, raw_paths_test_path, Y_train, Y_test
    
from keras_preprocessing.sequence import pad_sequences

def create_word_index_table(vocab):
    """
    Creating word to index table
    Input:
    vocab: list. The list of the node vocabulary
    """
    ixtoword = {}
    ixtoword[0] = 'END'
    ixtoword[1] = 'UNK'
    wordtoix = {}
    wordtoix['END'] = 0
    wordtoix['UNK'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return wordtoix, ixtoword


def encode_paths(raw_paths, wordtoix, MAX_SUBTREE_LENGTH):
    config = Config()
    encoded_paths = []
    encoded_paths_padded = []
    for problem in raw_paths:
        encoded_path = []
        for subtree in problem:
            encoded_subtree = []
            for p in subtree:
                if p in wordtoix:
                    encoded_subtree.append(wordtoix[p])
                else:
                    encoded_subtree.append(wordtoix['UNK'])
            encoded_path.append(encoded_subtree)
        encoded_path_padded = pad_sequences(encoded_path, maxlen=MAX_SUBTREE_LENGTH, padding='pre', truncating='post').tolist()
        if len(problem) < MAX_SUBTREE_LENGTH:
            [encoded_path_padded.append([0]*MAX_SUBTREE_LENGTH) for i in range(MAX_SUBTREE_LENGTH - len(problem))]
        else:
            encoded_path_padded = encoded_path_padded[:MAX_SUBTREE_LENGTH]
        encoded_paths.append(encoded_path)
        encoded_paths_padded.append(encoded_path_padded)
    return encoded_paths, encoded_paths_padded
                    

def preprocess_raw_paths_wordtoken(raw_paths, MAX_SUBTREE_LENGTH):
    path_hist = {}
    for problem in raw_paths:
        for subtree in problem:
            for p in subtree:
                if not p in path_hist:
                    path_hist[p] = 1
                else:
                    path_hist[p] += 1
    node_count = len(path_hist)
    valid_nodes = [path for path, count in path_hist.items()]
    
    wordtoix, ixtoword = create_word_index_table(valid_nodes)
    
    encoded_paths, encoded_paths_padded = encode_paths(raw_paths, wordtoix, MAX_SUBTREE_LENGTH)
    return encoded_paths, encoded_paths_padded, valid_nodes, wordtoix, ixtoword
    
    
from keras_preprocessing.sequence import pad_sequences


def encode_paths_path(raw_paths, wordtoix, MAX_SUBTREE_LENGTH):
    config = Config()
    encoded_paths = []
    encoded_paths_padded = []
    for problem in raw_paths:
        encoded_path = []
        encoded_path_padded = []
        for subtree in problem:
            if subtree in wordtoix:
                encoded_path.append(wordtoix[subtree])
                encoded_path_padded.append(wordtoix[subtree])
            else:
                encoded_path.append(wordtoix['UNK'])
                encoded_path_padded.append(wordtoix['UNK'])
        #encoded_path_padded = pad_sequences(encoded_path, maxlen=MAX_SUBTREE_LENGTH, padding='pre', truncating='post').tolist()
        if len(problem) < MAX_SUBTREE_LENGTH:
            [encoded_path_padded.append(0) for i in range(MAX_SUBTREE_LENGTH - len(problem))]
        else:
            encoded_path_padded = encoded_path_padded[:MAX_SUBTREE_LENGTH]
        encoded_paths.append(encoded_path)
        encoded_paths_padded.append(encoded_path_padded)
    return encoded_paths, encoded_paths_padded
                    

def preprocess_raw_paths_pathtoken(raw_paths, MAX_SUBTREE_LENGTH):
    path_hist = {}
    for problem in raw_paths:
        for p in problem:
            if not p in path_hist:
                path_hist[p] = 1
            else:
                path_hist[p] += 1
    node_count = len(path_hist)
    valid_nodes = [path for path, count in path_hist.items()]
    
    wordtoix, ixtoword = create_word_index_table(valid_nodes)
    
    encoded_paths, encoded_paths_padded = encode_paths_path(raw_paths, wordtoix, MAX_SUBTREE_LENGTH)
    return encoded_paths, encoded_paths_padded, valid_nodes, wordtoix, ixtoword
    
    
from gensim.models import Word2Vec

def embedding_paths(all_subtrees, vocab, embedding_size, wordtoix):
    VOCABULARY_SIZE = len(vocab) + 2
    embedding_weights = np.zeros((VOCABULARY_SIZE, embedding_size))
    #word2vec = Word2Vec(all_subtrees, size=embedding_size, window=5, workers=3, min_count=1)
    #for word, index in wordtoix.items():
    #  try:
    #    embedding_weights[index, :] = word2vec[word]
    #  except KeyError:
    #    pass
    return embedding_weights
    
    
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, concatenate, Embedding, Concatenate, Dropout, TimeDistributed, Softmax
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN, Conv1D
from tensorflow import keras
import tensorflow as tf
from keras import callbacks
import tensorflow.keras.backend as K
from config import Config

def create_model(embedding_weights, node_vocab_size, path_vocab_size, MAX_SUBTREE_LENGTH):
    config = Config()
    node_input = Input((MAX_SUBTREE_LENGTH,MAX_SUBTREE_LENGTH), dtype=tf.int32)
    path_input = Input((MAX_SUBTREE_LENGTH,), dtype=tf.int32)
    
    #embedding layer
    nodes_embedded = Embedding(node_vocab_size+2, config.embedding_size, trainable = True, name='node_embedding')(node_input)
    path_embedded = Embedding(path_vocab_size+2, config.embedding_size, 
                              trainable = True, name='path_embedding')(path_input) #(b,max_subtree,embedsize)
    
    # path embeddings from node embeddings
    nodes_embedded_merged = K.sum(nodes_embedded, axis=2) #(b,max_subtree,embedsize)
    
    # Attention Layer for node embeddings
    #node_attention_vectors = Dense(1,)(nodes_embedded)
    #node_attention_weights = Softmax()(node_attention_vectors)
    #nodes_embedded_merged = K.sum(nodes_embedded * node_attention_weights, axis=2)
    
    node_path_merged = concatenate([nodes_embedded_merged, path_embedded])
    
    subtree_vectors = TimeDistributed(Dense(config.embedding_size*2, use_bias=False, activation='tanh'))(node_path_merged)
    
    
    # Attention Layer
    attention_vectors = Dense(1, name='attention_vectors')(subtree_vectors)
    attention_weights = Softmax(axis=1, name='attention_weights')(attention_vectors)
    
    # Generating code vectors
    code_vectors = K.sum(subtree_vectors * attention_weights, axis=1)
    
    # Prediction layer
    output_class = Dense(config.num_classes, use_bias=False, activation='softmax')(code_vectors)
    
    model = Model(inputs=[node_input, path_input], outputs=output_class)
    return model
    
def train_evaluate(ga_individual_solution):   
    config = Config()
    data_path = DEFAULT_DATA_FILE
    codes, Y = read_data_GA(data_path)
    
    # Decode GA solution to integer for window_size and num_units
    max_depth_bits = BitArray(ga_individual_solution[0:3]) 
    max_depth = max_depth_bits.uint
    max_subtree_length_bits = BitArray(ga_individual_solution[3:]) 
    max_subtree_length = max_subtree_length_bits.uint
    print("************************************")
    print('Max depth: ', max_depth)
    print('Max subtree: ', max_subtree_length)
    print("************************************")
    
    # Return fitness score of 0 if max_depth is zero
    if max_depth == 0:
        return 0, 
    
    if max_subtree_length<64:
        return 0,
    
    # parse codes according to new max_depth
    main_df = parser_main(codes, Y, max_depth)
    
    # Segment the train_data based on new max_depth; split into train and validation (80/20)
    raw_paths_train, raw_paths_train_path, raw_paths_test, raw_paths_test_path, Y_train, Y_test = read_data(main_df)
    
    print("----Encoding start----")
    
    #encoding node
    encoded_paths_train, X_train, node_vocab, wordtoix, ixtoword = preprocess_raw_paths_wordtoken(raw_paths_train, 
                                                                                                  max_subtree_length)
    encoded_paths_test, X_test = encode_paths(raw_paths_test, wordtoix, max_subtree_length)
    
    #encoding path
    encoded_paths_train_path, X_train_path, path_vocab, pathtoix, ixtopath = preprocess_raw_paths_pathtoken(
        raw_paths_train_path, max_subtree_length)
    encoded_paths_test_path, X_test_path = encode_paths_path(raw_paths_test_path, pathtoix, max_subtree_length)
    print("----Encoding finished----")
    
    print("----Embedding start----")
    
    # word2vec weights calculation
    all_subtrees_train = []
    subtree_length_count = []
    for problem in raw_paths_train:
        for subtree in problem:
            all_subtrees_train.append(subtree)
            subtree_length_count.append(len(subtree))

    
    """
    Need to correctly implement word2vec
    """
    embedding_weights = embedding_paths(all_subtrees_train, node_vocab, config.embedding_size, wordtoix)
    print("----Embedding finished----")
    
    X_train = np.array(X_train)
    X_train_path = np.array(X_train_path)
    X_test = np.array(X_test)
    X_test_path = np.array(X_test_path)
    #print(Y_train.count(0), Y_train.count(1), Y_test.count(0), Y_test.count(1))
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    print("Train shape: Node: ", X_train.shape, " Path: ", X_train_path.shape, " Y: ", Y_train.shape)
    print("Test shape: Node: ", X_test.shape, " Path: ", X_test_path.shape, " Y: ", Y_test.shape)
    
    model = create_model(embedding_weights, len(node_vocab), len(path_vocab), max_subtree_length)
    #compile model
    model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
                           metrics=['acc'])
    # check summary of model
    #model.summary()
    
    # Early stopping
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 10, 
                                        restore_best_weights = True)
    
    # train model
    model.fit(x=[X_train,X_train_path], y=Y_train,batch_size=config.batch_size,epochs=config.epoch, 
              validation_data=([X_test,X_test_path], Y_test), callbacks =[earlystopping])
    
    # Performance
    # This version of the code does not have test set split. Using the validation data. Please update accordingly
    predicted = model.predict(x=[X_test,X_test_path])
    #print(predicted[0], Y_test[0])
    predicted = np.where(predicted > 0.5, 1, 0)
    #print(predicted[0], Y_test[0])
    accuracy = metrics.accuracy_score(Y_test, predicted)
    print("Acuracy: ", accuracy)
    return round(accuracy, 4),
    
def train_test_evaluate(MAX_DEPTH, MAX_SUBTREE_LENGTH):   
    config = Config()
    data_path = DEFAULT_DATA_FILE
    codes, Y = load_codes_and_scores(data_path)
    
    max_depth = MAX_DEPTH
    
    # parse codes according to new max_depth
    main_df = parser_main(codes, Y, max_depth)
    
    # Segment the train_data based on new max_depth; split into train and validation (80/20)
    raw_paths_train, raw_paths_train_path, raw_paths_test, raw_paths_test_path, Y_train, Y_test = read_data(main_df)
    
    print("----Encoding start----")
    
    #encoding node
    encoded_paths_train, X_train, node_vocab, wordtoix, ixtoword = preprocess_raw_paths_wordtoken(raw_paths_train, 
                                                                                                  MAX_SUBTREE_LENGTH)
    encoded_paths_test, X_test = encode_paths(raw_paths_test, wordtoix, MAX_SUBTREE_LENGTH)
    
    #encoding path
    encoded_paths_train_path, X_train_path, path_vocab, pathtoix, ixtopath = preprocess_raw_paths_pathtoken(
        raw_paths_train_path, MAX_SUBTREE_LENGTH)
    encoded_paths_test_path, X_test_path = encode_paths_path(raw_paths_test_path, pathtoix, MAX_SUBTREE_LENGTH)
    print("----Encoding finished----")
    
    print("----Embedding start----")
    
    # word2vec weights calculation
    all_subtrees_train = []
    subtree_length_count = []
    for problem in raw_paths_train:
        for subtree in problem:
            all_subtrees_train.append(subtree)
            subtree_length_count.append(len(subtree))
            
    print("Maximum number of subtree: ", max(len(x) for x in raw_paths_train))
    print("Maximum subtree length: ", max(subtree_length_count))
    
    """
    Need to correctly implement word2vec
    """
    embedding_weights = embedding_paths(all_subtrees_train, node_vocab, config.embedding_size, wordtoix)
    print("----Embedding finished----")
    
    X_train = np.array(X_train)
    X_train_path = np.array(X_train_path)
    X_test = np.array(X_test)
    X_test_path = np.array(X_test_path)
    #print(Y_train.count(0), Y_train.count(1), Y_test.count(0), Y_test.count(1))
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    print("Train shape: Node: ", X_train.shape, " Path: ", X_train_path.shape, " Y: ", Y_train.shape)
    print("Test shape: Node: ", X_test.shape, " Path: ", X_test_path.shape, " Y: ", Y_test.shape)
    
    model = create_model(embedding_weights, len(node_vocab), len(path_vocab), MAX_SUBTREE_LENGTH)
    #compile model
    model.compile(loss='categorical_crossentropy',
                           optimizer='adamax',
                           metrics=['acc'])
    # check summary of model
    model.summary()
    
    # Early stopping
    earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 20, 
                                        restore_best_weights = True)
    
    y_train_labels = np.argmax(Y_train, axis=1)
    unique_classes = np.unique(y_train_labels)
    cw_vals = compute_class_weight("balanced", classes=unique_classes, y=y_train_labels)
    class_weight = {int(c): float(w) for c, w in zip(unique_classes, cw_vals)}

    # train model
    history = model.fit(
        x=[X_train, X_train_path],
        y=Y_train,
        batch_size=32,
        epochs=config.epoch,
        validation_data=([X_test, X_test_path], Y_test),
        callbacks=[earlystopping],
        class_weight=class_weight,
    )

    # Evaluate on validation/test split
    predicted_probs = model.predict(x=[X_test, X_test_path])
    y_true_cls = np.argmax(Y_test, axis=1)
    y_pred_cls = np.argmax(predicted_probs, axis=1)
    accuracy = float(metrics.accuracy_score(y_true_cls, y_pred_cls))
    precision = float(metrics.precision_score(y_true_cls, y_pred_cls, zero_division=0))
    recall = float(metrics.recall_score(y_true_cls, y_pred_cls, zero_division=0))
    f1 = float(metrics.f1_score(y_true_cls, y_pred_cls, zero_division=0))
    print("Final Validation Accuracy: ", round(accuracy, 4))

    metadata = {
        "max_depth": int(MAX_DEPTH),
        "max_subtree_length": int(MAX_SUBTREE_LENGTH),
        "node_vocab_size": len(node_vocab),
        "path_vocab_size": len(path_vocab),
        "embedding_size": config.embedding_size,
        "num_classes": config.num_classes,
        "leaf_flag": config.leaf_flag,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_test)),
        "total_compilable_samples": int(len(main_df)),
        "val_accuracy": round(accuracy, 4),
        "val_precision": round(precision, 4),
        "val_recall": round(recall, 4),
        "val_f1": round(f1, 4),
        "epochs_trained": len(history.history.get("loss", [])),
        "wordtoix": wordtoix,
        "ixtoword": {str(k): v for k, v in ixtoword.items()},
        "pathtoix": pathtoix,
        "ixtopath": {str(k): v for k, v in ixtopath.items()},
    }

    training_summary = {
        "dataset_source": "All 9 Week_01-02 classes in Analyze data/clean (excluding Monitored_Codes_Dipantau.csv)",
        "max_depth": int(MAX_DEPTH),
        "max_subtree_length": int(MAX_SUBTREE_LENGTH),
        "node_vocab_size": len(node_vocab),
        "path_vocab_size": len(path_vocab),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_test)),
        "total_compilable_samples": int(len(main_df)),
        "val_accuracy": round(accuracy, 4),
        "val_precision": round(precision, 4),
        "val_recall": round(recall, 4),
        "val_f1": round(f1, 4),
        "epochs_trained": len(history.history.get("loss", [])),
        "history": {
            k: [round(float(v), 5) for v in vals]
            for k, vals in history.history.items()
        },
    }

    # Save trained model, weights, and vocabulary metadata to both saved_model_dap and saved_model
    for folder_name in ("saved_model_dap", "saved_model"):
        save_dir = Path(__file__).resolve().parent.parent / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        weights_path = save_dir / "sann_dap_weights.h5"
        model_path = save_dir / "sann_dap_model.h5"
        metadata_path = save_dir / "metadata.json"
        summary_path = save_dir / "training_summary.json"

        model.save_weights(str(weights_path))
        try:
            model.save(str(model_path))
        except Exception:
            pass

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)

        print(f"Model and metadata saved to: {save_dir}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train SANN-DAP on Week 1 dataset excluding Monitored_Codes_Dipantau.csv")
    parser.add_argument("--skip-ga", action="store_true", help="Skip GA search and train directly with specified max_depth and max_subtree_length")
    parser.add_argument("--max-depth", type=int, default=3, help="Max AST subtree depth (default: 3)")
    parser.add_argument("--max-subtree-length", type=int, default=64, help="Max subtree sequence length (default: 64)")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Force rebuild combined Week 1 training CSV")
    args = parser.parse_args()

    if args.rebuild_dataset:
        build_week1_train_dataset(force_rebuild=True)

    if args.skip_ga:
        best_max_depth = args.max_depth
        best_max_subtree_length = args.max_subtree_length
    else:
        population_size = 10
        num_generations = 5
        gene_length = 10
        start = time()
        creator.create('FitnessMax', base.Fitness, weights = (1.0,))
        creator.create('Individual', list , fitness = creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register('binary', bernoulli.rvs, 0.5)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
        toolbox.register('population', tools.initRepeat, list , toolbox.individual)

        toolbox.register('mate', tools.cxOrdered)
        toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
        toolbox.register('select', tools.selRoulette)
        toolbox.register('evaluate', train_evaluate)

        population = toolbox.population(n = population_size)
        r = algorithms.eaSimple(population, toolbox, cxpb = 0.6, mutpb = 0.2, ngen = num_generations, verbose = False)
        
        best_individuals = tools.selBest(population, k = 1)
        for bi in best_individuals:
            max_depth_bits = BitArray(bi[0:3])
            best_max_depth = max_depth_bits.uint
            max_subtree_length_bits = BitArray(bi[3:])
            best_max_subtree_length = max_subtree_length_bits.uint
            print('Best Max depth: ', best_max_depth)
            print('Best Max subtree length: ', best_max_subtree_length)

    print("************************************")
    print("Final Training")
    print("************************************")
    train_test_evaluate(best_max_depth, best_max_subtree_length)


if __name__ == "__main__":
    main()


