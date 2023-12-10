import os
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import NMF
import joblib

class TopicModeller():

    def __init__(self, topic_map, save_root):
        self.topic_map = topic_map
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.save_root = save_root
        
    def create_topic_tfidf(self):
        tokenized_chunks = [self.tokenizer(chunk, return_tensors='pt', padding=True, truncation=True) for chunk in tqdm(self.topic_map['topic_text'])]
        # Use BERT to get embeddings for each chunk
        chunk_embeddings = [self.bert_model(**tokens)['pooler_output'].detach().numpy() for tokens in tqdm(tokenized_chunks)]
        doc_matrix = np.vstack(chunk_embeddings)
        doc_matrix_tfidf = self.vectorizer.fit_transform(self.topic_map['topic_text']).toarray()
        return doc_matrix_tfidf

    def create_nmf_model(self, doc_matrix_tfidf):
        num_topics = len(self.topic_map.topic_name.unique())  # Specify the number of topics
        nmf_model = NMF(n_components=num_topics, random_state=42)
        nmf_model.fit(doc_matrix_tfidf)
        # Extract topic-word and document-topic matrices
        topic_word_matrix = torch.from_numpy(nmf_model.components_)
        doc_topic_matrix = torch.from_numpy(nmf_model.transform(doc_matrix_tfidf))
        topic_assignments = torch.argmax(doc_topic_matrix, dim=1)
        return topic_word_matrix, doc_topic_matrix, topic_assignments

    def process(self):
        doc_matrix_tfidf = self.create_topic_tfidf()
        topic_word_matrix, doc_topic_matrix, topic_assignments = self.create_nmf_model(doc_matrix_tfidf)
        self.topic_map['topic_predicted'] = topic_assignments.numpy()
        self.topic_map['topic_predicted'] = self.topic_map['topic_predicted'].map(lambda idx: self.topic_map['topic_name'][idx])
        joblib.dump(self.vectorizer, os.path.join(self.save_root, 'text_vectorizer.joblib'))
        np.save(os.path.join(self.save_root, 'topic_word_matrix.npy'), topic_word_matrix)
        np.save(os.path.join(self.save_root, 'doc_topic_matrix.npy'), doc_topic_matrix)
        np.save(os.path.join(self.save_root, 'doc_matrix_tfidf.npy'), doc_matrix_tfidf)
        self.topic_map.to_csv(os.path.join(self.save_root, 'topic_map.csv'), index=None)