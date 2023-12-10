import os

class CreateKG():

    def __init__(self, doc_matrix_tfidf, topic_map, save_root):
        self.doc_matrix_tfidf = doc_matrix_tfidf
        self.topic_map = topic_map
        self.label_encoder = LabelEncoder()
        self.topic_map['topic_predicted_encoded'] = self.label_encoder.fit_transform(topic_map['topic_predicted'])
        # Create a knowledge graph using networkx
        self.topic_graph = nx.DiGraph()
        self.ner_model = spacy.load("en_core_web_sm")
        self.save_root = save_root

    def create_nodes(self):
        # Add nodes and set topic predicted as node attribute
        for index, row in tqdm(self.topic_map.iterrows()):
            text = row['topic_text']
            topic_predicted = row['topic_predicted_encoded']
            
            nodes = [node.text for node in self.ner_model(text).ents]
            self.topic_graph.add_nodes_from(nodes, topic_predicted=topic_predicted)

    def create_weighted_edges(self):
        # Add weighted edges with 'Topic Predicted' as weights
        for index, row in tqdm(self.topic_map.iterrows()):
            text = row['topic_text']
            nodes = [node.text for node in self.ner_model(text).ents]
            topic_predicted = row['topic_predicted_encoded']
            
            # Add edges without using dictionary format
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    self.topic_graph.add_edge(nodes[i], nodes[j], weight=topic_predicted)

    def create_torch_graph(self):
        X = self.doc_matrix_tfidf.copy()
        Y = torch.tensor(self.topic_map['topic_predicted_encoded'].values, dtype=torch.long)
        # Get edges from the graph and vectorize them
        edges = [' '.join(map(str, edge)) for edge in self.topic_graph.edges]
        vectorizer_edges = CountVectorizer()
        X_edges = vectorizer_edges.fit_transform(edges).toarray()
        # Convert edges to PyTorch LongTensor
        edge_index = torch.tensor(X_edges, dtype=torch.long).t().contiguous()
        edge_index = edge_index.view(2, -1)
        # Create a PyTorch Geometric Data object
        data = Data(x=torch.tensor(X, dtype=torch.float32), y=Y, edge_index=edge_index)
        return vectorizer_edges, data

    def process(self):
        self.create_nodes()
        self.create_weighted_edges()
        vectorizer_edges, data = self.create_torch_graph()
        self.topic_map.to_csv(os.path.join(self.save_root, 'topic_map.csv'), index=None)
        joblib.dump(vectorizer_edges, os.path.join(self.save_root, 'edges_vectorizer.joblib'))
        nx.write_gpickle(self.topic_graph, os.path.join(self.save_root, 'topic_graph.pkl'))
        torch.save(data, os.path.join(self.save_root, 'topic_graph.pth'))
        joblib.dump(self.label_encoder, os.path.join(self.save_root, 'label_encoder.joblib'))