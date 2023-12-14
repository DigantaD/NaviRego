# NaviRego
Marine Regulations - Topic Modelling &amp; Information Retrieval

# The Problem Statement
This problem statement is aimed at creating an NLP infrastructure which can help us pinpoint to a definitive topic on Marine Regulations based on a text sample provided and retrieving relevant detailed information in an explained and simplified format for either an SME or even a normal person to understand.

# About the Data
The data consists of a single or multiple text files which can also be interpreted as a Vector Text Database containing paragraphs and sections on Marine Regulations, in this case, the Solas Text File

# Solutioning Approach
Following are the steps that have been taken to approach the solution to the problem statement:
* Chunking of the text files into logical sections
* Retrieval of possible Topic Names using ChatGPT-3.5
* Narrowing down the Topic Categories using Non-Negative Matrix Factorization
* Extraction of Entities from each section-wise text
* Creation of a Topic Modelling focussed Knowledge Graph
* Conversion to a Trainable Knowledge Graph Architecture
* Model training & Optimizations
* Retrieval of Topic Category from a singular or multiple texts
* Usage of the particular Topic nodes and edges and the input text to extract simplified information using ChatGPT-3.5

# Concepts and Frameworks used
* OpenAI ChatGPT-3.5-turbo-16K
* Spacy
* Pandas
* Numpy
* Pytorch
* Torch Geometric
* NetworkX
* Transformers
* Scikit-Learn
* Matplotlib

# Models Used
* BertTokenizer
* Bert uncased
* Non-Negative Matrix Factorization
* GCNConv
* TFIDF Vectorizer
* Count Vectorizer

# Outputs & Related Figures
## Sample Topics from NMF: 
['Auxiliary Steering Gear', 'Translation Requirement', 'Marine Regulations Amendment Procedures', 'Stepped Bulkhead Deck', 'Marine Regulations Supremacy', 'Marine Regulations Overview', 'Marine Certificate Validity', 'Marine Certificate Acceptance', 'Marine Regulations: Inflammable vs Flammable', 'Marine Safety Regulations', 'Marine Regulation Exemptions', 'Marine Regulations for Machinery Spaces', 'Marine Regulations: Ratification and Deposit of Instruments','Marine Regulations: Amendment Acceptance and Objection Procedures', 'Marine Convention Signature and Accession Dates', 'Marine Survey Endorsement', 'Amendments to the Convention', 'Marine Regulations for Ship Construction and Repairs', 'Marine Regulations: Continuation of Existing Treaties', 'Title: Exemption from Infringement']
## Top Words from Indexed Topics:
  	* Topic 1: of, the, date, present, deposit, items, inspection, 2020, bottom, do
  	* Topic 2: 07, 2022, 05, updated, imo, solas, maritime, chapter, international, organization
  	* Topic 3: entry, into, force, date, present, expanded, voting, determined, majority, thirds
  	* Topic 4: 25, keel, laid, stage, which, similar, is, ship, new, or
  	* Topic 5: certificate, existing, expiry, date, endorse, exceed, completed, placed, authorized, person
  	* Topic 6: signature, ratification, approval, accession, acceptance, and, followed, or, deposit, its
  	* Topic 7: ships, of, structure, expression, for, respectively, 2024, keel, constructed, keels
  	* Topic 8: certificates, contracting, purposes, covered, authority, 17, by, accepted, acceptance, government
  	* Topic 9: emergency, source, power, electrical, supply, failure, event, switchboard, intended, 12
  	* Topic 10: deposited, deemed, to, instrument, accepted, approval, ratification, amended, accession, apply
## Complete Knowledge Graph
![navirego_graph](https://github.com/DigantaD/NaviRego/assets/27140456/e9726adf-28af-4ead-b1e0-e498c4c0454d)
## Graph Node: Chemical
![navirego_node](https://github.com/DigantaD/NaviRego/assets/27140456/8b1b460d-69ec-4449-ae8e-819646ed712c)
## Graph Edge: Protocol
![navirego_edge](https://github.com/DigantaD/NaviRego/assets/27140456/7e19bec7-0fe9-42f8-a5a5-71143405ded7)

# Train Data & Model Performance
* Data
	* Train: Data(x=[268, 1000], edge_index=[2, 172665], y=[268])
   	* Validation: Data(x=[57, 1000], edge_index=[2, 172665], y=[57])
   	* Test: Data(x=[58, 1000], edge_index=[2, 172665], y=[58])
* Number of actual Chunks: 5400+
* Number of chunks trained: ~400
* Training Rounds: 10,000
* Performance: Epoch: 10000, Loss: 0.004, Acc: 1.0, Val Loss: 11.523, Val Acc: 0.246

# Limitations
* Limited access of GPT usage due to which not more than 500 chunks could be made at one time
* Less training data as a result
* Less Graph Complexity & Node-Edge Relations
* Flask API failing to function properly from a Windows backend

# Actual Modules
* data_creator.py - Creation of Topic Map
* topic_modeller.py - Categorization of Topics, Entity Extraction & Topic Modelling
* graph_creator.py - Knowledge Graph Creation
* trainer.py - Knowledge Graph Training
* info_retrieval.py - Code not available as prediction fails mostly due to very less trained data (Can update it if resources are available)
* app.py - Flask App

Thank you for this opportunity to showcase this very interesting project although not optimized and not fully complete but surely can discuss more algorithms out 1-on-1!

















