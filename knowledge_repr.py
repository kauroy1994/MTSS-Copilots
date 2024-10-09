import gradio as gr
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline
import os

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face pipelines
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
relation_extraction_model = pipeline("text2text-generation", model="t5-base")
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def spacy_based_relation_extraction(text):
    doc = nlp(text)
    relations = []
    
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subj = [child for child in token.lefts if child.dep_ in ("nsubj", "nsubjpass")]
                obj = [child for child in token.rights if child.dep_ in ("dobj", "pobj", "attr")]
                
                if subj and obj:
                    subj_text = " ".join([w.text for w in subj])
                    obj_text = " ".join([w.text for w in obj])
                    relations.append((subj_text, token.lemma_, obj_text))
    
    return relations

def huggingface_relation_extraction(text):
    ner_results = ner_pipeline(text)
    unique_entities = []
    for entity in ner_results:
        if entity['entity_group'] in ['PER', 'ORG', 'LOC']:
            if not any(ent['word'] == entity['word'] for ent in unique_entities):
                unique_entities.append(entity)
    
    relations = []
    for i, ent1 in enumerate(unique_entities):
        for j, ent2 in enumerate(unique_entities):
            if i != j:
                input_text = f"Extract the relation between {ent1['word']} and {ent2['word']} in this context: {text}"
                result = relation_extraction_model(input_text)
                relation = result[0]['generated_text']
                relations.append((ent1['word'], relation, ent2['word']))
                
    return relations

def visualize_relations(relations):
    G = nx.DiGraph()
    for subj, rel, obj in relations:
        G.add_edge(subj, obj, label=rel)
    
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Relation Extraction Graph")
    
    # Save the graph as an image file in the current directory
    image_path = "relation_graph.png"
    plt.savefig(image_path)
    plt.close()
    return image_path

def summarize_text(text):
    summary = summarization_model(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def generate_questions_and_answers(text):
    ner_results = ner_pipeline(text)
    entities = [ent['word'] for ent in ner_results if ent['entity_group'] == 'PER']
    
    if not entities:
        return "<div>No suitable entities found to generate questions.</div>"
    
    qa_html = "<table style='width:100%; border-collapse: collapse;'>"
    qa_html += "<tr style='background-color: #f2f2f2;'><th style='border: 1px solid black; padding: 8px;'>Question</th><th style='border: 1px solid black; padding: 8px;'>Answer</th></tr>"
    
    for entity in entities:
        question_text = f"What is the information about {entity} in this context?"
        result = qa_pipeline(question=question_text, context=text)
        qa_html += f"<tr><td style='border: 1px solid black; padding: 8px;'>{question_text}</td><td style='border: 1px solid black; padding: 8px;'>{result['answer']}</td></tr>"
    
    qa_html += "</table>"
    return qa_html

def process_text(text, approach):
    if approach == "SpaCy-based Relation Extraction":
        triples = spacy_based_relation_extraction(text)
        if triples:
            image_path = visualize_relations(triples)
            return image_path, "Graph generated for SpaCy-based Relation Extraction."
        else:
            return None, "No relations found in the text."
    elif approach == "Hugging Face Relation Extraction":
        triples = huggingface_relation_extraction(text)
        if triples:
            image_path = visualize_relations(triples)
            return image_path, "Graph generated for Hugging Face Relation Extraction."
        else:
            return None, "No relations found in the text."
    elif approach == "Summarization":
        summary = summarize_text(text)
        return None, summary
    elif approach == "Question and Answer":
        qa_html = generate_questions_and_answers(text)
        return None, qa_html

# Gradio Interface
def main_interface():
    gr.Interface(
        fn=process_text,
        inputs=[
            gr.Textbox(lines=10, placeholder="Enter case file text or any text...", label="Input Text"),
            gr.Dropdown(
                ["SpaCy-based Relation Extraction", "Hugging Face Relation Extraction", "Summarization", "Question and Answer"],
                label="Choose Representation Approach",
                value="SpaCy-based Relation Extraction"
            )
        ],
        outputs=[
            gr.Image(type="filepath", label="Relation Graph"),
            gr.HTML(label="Result"),
        ],
        title="Enhanced Knowledge Representation Comparison",
        description=(
            "Compare and contrast different knowledge representations of text. "
            "Select an approach (relation extraction, summarization, or QA) to see the output."
        ),
        theme="default",
        layout="vertical"
    ).launch()

# Run the interface
if __name__ == "__main__":
    main_interface()

