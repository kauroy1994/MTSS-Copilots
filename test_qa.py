import json
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import io
from typing import List, Tuple

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define similarity thresholds
NEURAL_SIMILARITY_THRESHOLD = 0.0  # Neural similarity threshold
TRIPLE_SIMILARITY_THRESHOLD = 0.0  # Knowledge graph (triple-based) similarity threshold


def load_json_data(file_path: str) -> List[dict]:
    """
    Load JSON data from the given file path.

    :param file_path: Path to the JSON file.
    :return: List of dictionaries containing question-answer data.
    :raises: FileNotFoundError, ValueError
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            assert isinstance(data, list), "JSON data should be a list of dictionaries."
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in the provided file.")
    except AssertionError as e:
        raise ValueError(f"Data assertion failed: {e}")


def extract_questions_and_answers(data: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Extract questions and answers from the provided JSON data.

    :param data: List of dictionaries containing question-answer pairs.
    :return: Tuple of lists (questions, answers).
    :raises: AssertionError if no questions or answers are found.
    """
    questions = []
    answers = []
    for item in data:
        if "question" in item and "answer" in item:
            questions.append(item['question'])
            answers.append(item['answer'])
    assert questions, "No questions found in the provided data."
    assert answers, "No answers found in the provided data."
    return questions, answers


def clean_questions(questions: List[str]) -> List[str]:
    """
    Clean the list of questions by removing empty or invalid strings.

    :param questions: List of questions.
    :return: Cleaned list of valid questions.
    """
    return [q.strip() for q in questions if isinstance(q, str) and q.strip()]


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract subject-verb-object triples (SVO) from the provided text.

    :param text: Input text for extracting triples.
    :return: List of tuples containing (subject, verb, object).
    """
    doc = nlp(text)
    triples = []

    for sent in doc.sents:
        subj, verb, obj = None, None, None
        for token in sent:
            if token.dep_ == "nsubj":  # Subject
                subj = token.text
            if token.dep_ == "ROOT":  # Main verb
                verb = token.lemma_  # Use the lemma of the verb
            if token.dep_ == "dobj":  # Direct object
                obj = token.text
        if subj and verb and obj:
            triples.append((subj, verb, obj))

    return triples


def triple_similarity(query_triples: List[Tuple[str, str, str]], answer_triples: List[Tuple[str, str, str]]) -> float:
    """
    Calculate the similarity between two lists of triples (subject-verb-object).

    :param query_triples: SVO triples extracted from the query.
    :param answer_triples: SVO triples extracted from an answer.
    :return: A similarity score between 0 and 1 based on matching triples.
    """
    matches = 0
    for query_triple in query_triples:
        if query_triple in answer_triples:
            matches += 1
    return matches / len(query_triples) if query_triples else 0.0


def format_knowledge_panel(triples: List[Tuple[str, str, str]]) -> str:
    """
    Format extracted triples into a knowledge panel.

    :param triples: List of triples (subject, verb, object).
    :return: Formatted string for the knowledge panel.
    """
    panel_text = "\n".join(f"**{subj}** {relation} **{obj}**" for subj, relation, obj in triples)
    return panel_text if panel_text else "No entities and relationships found."


def plot_top_matches(matched_questions: List[str], similarity_scores: np.ndarray) -> Image.Image:
    """
    Plot top matched questions and their similarity scores as a bar chart.

    :param matched_questions: List of matched questions.
    :param similarity_scores: Array of similarity scores for each question.
    :return: Bar chart as a PIL image.
    :raises: RuntimeError if there is an error generating the chart.
    """
    try:
        plt.figure(figsize=(8, 4))  # Adjust figure size
        y_pos = np.arange(len(matched_questions))
        plt.barh(y_pos, similarity_scores, align='center', alpha=0.7, color='blue')
        plt.yticks(y_pos, matched_questions, fontsize=10)
        plt.xlabel('Similarity Score', fontsize=12)
        plt.title('Top Matched Questions', fontsize=14)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return Image.open(buf)
    except Exception as e:
        raise RuntimeError(f"Error generating the top matched questions chart: {e}")


def search_and_visualize(query: str) -> Tuple[str, str, str, str, Image.Image, str]:
    """
    Handle the query search and visualization of results with a filtering mechanism.

    :param query: The search query input by the user.
    :return: Tuple containing matched question, answer, knowledge panels, top matches image, and suggested queries.
    :raises: RuntimeError if there is an error processing the query.
    """
    try:
        # Step 1: Perform the neural embedding search
        query_embedding = model.encode(query, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, question_embeddings)[0]

        # Step 2: Rank questions by similarity
        ranked_idx = np.argsort(similarities.cpu().numpy())[::-1].copy()
        top_idx = ranked_idx[:5].copy()

        matched_questions = [questions[idx] for idx in top_idx]
        matched_answers = [answers[idx] for idx in top_idx]
        similarity_scores = similarities[top_idx].cpu().numpy().copy()

        # Check if the highest similarity score is above the threshold
        if similarity_scores[0] < NEURAL_SIMILARITY_THRESHOLD:
            return "Sorry, we don't have sufficient information to answer this query.", "", "", "", None, '\n'.join(questions)

        # Step 3: Extract triples for the top answer and the query
        query_triples = extract_triples(query)
        answer_triples = extract_triples(matched_answers[0])

        # Calculate knowledge-graph-based similarity
        triple_sim = triple_similarity(query_triples, answer_triples)

        # If the triple similarity is below the threshold, return boilerplate response
        if triple_sim < TRIPLE_SIMILARITY_THRESHOLD:
            return "Sorry, we don't have sufficient information to answer this query.", "", "", "", None, '\n'.join(questions)

        # Step 4: Format knowledge panels
        query_knowledge_panel = format_knowledge_panel(query_triples)
        answer_knowledge_panel = format_knowledge_panel(answer_triples)

        # Step 5: Plot top matched questions and similarity scores
        top_matches_img = plot_top_matches(matched_questions, similarity_scores)

        # Step 6: Return the matched question, answer, knowledge panels, and suggested queries
        return matched_questions[0], matched_answers[0], query_knowledge_panel, answer_knowledge_panel, top_matches_img, '\n'.join(questions)

    except Exception as e:
        raise RuntimeError(f"Error processing the search query: {e}")


# Load and prepare data
qa_data = load_json_data('admin_qa.json')
questions, answers = extract_questions_and_answers(qa_data)

# Clean the questions to remove invalid or empty values
cleaned_questions = clean_questions(questions)

# Make sure we have valid questions to work with
assert cleaned_questions, "No valid questions to encode."

# Embed cleaned questions
question_embeddings = model.encode(cleaned_questions, convert_to_tensor=True)

# Select the first valid question as the placeholder
placeholder_question: str = cleaned_questions[0] if cleaned_questions else "Enter your query here..."


def gradio_interface(query: str) -> Tuple[str, str, str, str, Image.Image, str]:
    """
    Interface function for Gradio. It handles the user's query and returns the results.

    :param query: User's query input.
    :return: Results to be displayed on the Gradio interface.
    """
    return search_and_visualize(query)


# Gradio Interface Setup
with gr.Blocks() as demo:
    gr.Markdown("# Question-Answer Knowledge Search System")

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="Enter your query", placeholder=placeholder_question)
            submit_btn = gr.Button("Search")

        with gr.Column(scale=2):
            matched_question_output = gr.Textbox(label="Matched Question", interactive=False)
            answer_output = gr.Textbox(label="Answer", interactive=False)

    gr.Markdown("## Knowledge Panel (Query & Answer)")
    with gr.Row():
        with gr.Column(scale=1):
            query_panel_output = gr.Markdown(label="Query Knowledge Panel")
        with gr.Column(scale=1):
            answer_panel_output = gr.Markdown(label="Answer Knowledge Panel")

    gr.Markdown("## Top Matched Questions and Similarity Scores")
    top_matches_output = gr.Image(label="Top Matched Questions (with Similarity Scores)", type="pil")

    gr.Markdown("## Query Suggestions")
    suggested_queries_output = gr.Textbox(label="Suggested Queries", interactive=False)

    submit_btn.click(gradio_interface, query_input, [matched_question_output, answer_output, query_panel_output, answer_panel_output, top_matches_output, suggested_queries_output])

# Launch the Gradio App
demo.launch()
