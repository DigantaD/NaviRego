import os
import openai
openai.api_key = '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'

class CreateData():

    def __init__(self, text_file_path, save_root):
        self.text_file_path = text_file_path
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2000000
        self.ner_model = spacy.load("en_core_web_sm")
        self.save_root = save_root

    def chunk_into_paragraphs(self, text):
        # Process the text using spaCy
        doc = self.nlp(text)
        # Extract paragraphs
        paragraphs = [par.text.strip() for par in doc.sents]
        return paragraphs

    def generate_topic_name(self, metadata):
        prompt = f"""
            I am sending you a chunk of text from a text document.
            The chunk is related to Marine Regulations
            Analyze the text and give an appropriate title to it in response. 
            The title should be short and crisp and no detail.
            Be as quick as possible.
            Topic Textual Content: {metadata}
        """
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            temperature=0.5,
            top_p=0.5,
            frequency_penalty=0.5,
            messages= [
                {
                    "role": "system",
                    "content": "You are Topic Title Giver",
                },
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
            ],
        )
        response = res["choices"][0]["message"]["content"]
        return response

    def process(self):
        with open(self.text_file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
        # Chunk the document into paragraphs
        paragraphs = self.chunk_into_paragraphs(document_text)
        select_paragraphs = paragraphs[:1000]
        topic_map = {'topic_name': list(), 'topic_text': list()}
        for i, paragraph in tqdm(enumerate(select_paragraphs, start=1)):
            title = self.generate_topic_name(paragraph)
            topic_map['topic_name'].append(title.strip())
            topic_map['topic_text'].append(paragraph)
        topic_map = pd.DataFrame(topic_map)
        topic_map.to_csv(os.path.join(self.save_root, 'topic_map.csv'), index_col=False)