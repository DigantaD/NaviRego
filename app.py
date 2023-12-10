import os
from flask import Flask, jsonify, request
import requests
from flask_cors import CORS 
from decouple import config
from navi_topic_modeller.data_creator import CreateData
from navi_topic_modeller.topic_modeller import TopicModeller
from navi_topic_modeller.graph_creator import CreateKG
from navi_topic_modeller.trainer import TopicGraphModel, TrainerNetwork

app = Flask(__name__)
CORS(app, origins=config('CORS_ALLOWED_ORIGINS', default='*'))

if os.path.isdir('metadata'):
    pass
else:
    os.mkdir('metadata')

@app.route('/trainNewVectors', methods=['POST'])
def train():
    save_root = 'metadata'
    text_file_path = request.json.get('text_file_path')
    data_creator_object = CreateData(text_file_path, save_root)
    data_creator_object.process()
    topic_map = pd.read_csv(os.path.join(save_root, 'topic_map.csv'), index_col=False)
    modeller_object = TopicModeller(topic_map, save_root)
    modeller_object.process()
    topic_map = pd.read_csv(os.path.join(save_root, 'topic_map.csv'), index_col=False)
    text_vectorizer = joblib.load(os.path.join(save_root), 'text_vectorizer.joblib')
    topic_word_matrix = np.load(os.path.join(save_root, 'topic_word_matrix.npy'))
    doc_topic_matrix = np.load(os.path.join(save_root, 'doc_topic_matrix.npy'))
    doc_matrix_tfidf = np.load(os.path.join(save_root, 'doc_matric_tfidf.npy'))
    graph_object = CreateKG(doc_matric_tfidf, topic_map, save_root)
    graph_object.process()
    topic_map = pd.read_csv(os.path.join(save_root, 'topic_map.csv'), index_col=False)
    edges_vectorizer = joblib.load(os.path.join(save_root), 'edges_vectorizer.joblib')
    topic_graph = nx.read_gpickle(os.path.join(save_root, 'topic_graph.pkl'))
    topic_graph_torch = torch.load(os.path.join(save_root, 'topic_graph.pth'))
    label_encoder = joblib.load(os.path.join(save_root, 'label_encoder.joblib'))
    trainer_object = TrainerNetwork(topic_graph_torch, label_encoder)
    best_model_path = trainer_object.process()

    return jsonify({
        'training_status': True
    })