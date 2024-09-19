import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from typing import List, Dict

# Load the tokenizer and model for question generation (fine-tuned for QG)
tokenizer_qg = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
model_qg = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")

# Load the question-answering model
qa_pipeline = pipeline("question-answering")

def split_into_sentences(text: str) -> List[str]:
    """
    Split the text into individual sentences using regular expressions.

    Args:
        text (str): The input text to split.

    Returns:
        List[str]: A list of sentences.
    """
    # Simple regex to split sentences by punctuation (., ?, !)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return sentences

def generate_questions(text: str) -> List[str]:
    """
    Generate meaningful questions from the input text.

    Args:
        text (str): The input text to generate questions from.

    Returns:
        List[str]: A list of generated questions.
    """
    # Tokenize the input text for question generation
    input_ids = tokenizer_qg.encode("generate question: " + text, return_tensors="pt")

    # Generate questions with beam search for diversity
    outputs = model_qg.generate(input_ids, max_length=64, num_beams=5, early_stopping=True)
    
    # Decode the generated questions
    questions = [tokenizer_qg.decode(output, skip_special_tokens=True) for output in outputs]

    # Filter for valid questions
    filtered_questions = [q.strip() for q in questions 
                          if len(q.split()) > 5 and q.endswith("?") 
                          and (len(text.split()) > 0 and not q.lower().startswith(text.lower().split()[0]))]

    return filtered_questions

def generate_questions_for_text(text: str) -> List[str]:
    """
    Split the text into smaller chunks (sentences) and generate questions for each.

    Args:
        text (str): The input text to generate questions from.

    Returns:
        List[str]: A list of generated questions from the entire text.
    """
    sentences = split_into_sentences(text)
    all_questions = []

    # Generate questions for each sentence or chunk of text
    for sentence in sentences:
        if sentence.strip():  # Ensure the sentence is not empty
            questions = generate_questions(sentence)
            all_questions.extend(questions)

    return all_questions

def generate_answers(questions: List[str], context: str) -> List[Dict[str, str]]:
    """
    Generate answers for the list of questions using the provided context.

    Args:
        questions (List[str]): The list of generated questions.
        context (str): The context from which to extract answers.

    Returns:
        List[Dict]: A list of question-answer pairs.
    """
    qas = []
    for question in questions:
        answer = qa_pipeline(question=question, context=context)
        qas.append({
            'question': question,
            'answer': answer['answer']
        })
    
    return qas

# Example text (replace with your actual text)
example_text = """
John Doe is a 12-year-old student in the seventh grade. Recently, he has been experiencing difficulties in school,
    both academically and socially. His grades have started to decline, particularly in mathematics and science.
    Teachers have noted that he seems disengaged during lessons and is frequently distracted. Additionally,
    John has had several conflicts with his peers, including incidents of verbal arguments and one physical altercation.
    His parents have reported that he is increasingly anxious and reluctant to attend school. They have also observed
    changes in his behavior at home, including irritability and withdrawal from family activities.

    After meeting with John's parents, the school counselor suggested that these issues might be linked to the recent
    move the family made to a new neighborhood. John has struggled to adjust to his new environment and has expressed
    feelings of loneliness and isolation. The counselor recommended that John participate in social skills groups
    and be provided with additional academic support. They also discussed the possibility of further psychological
    evaluation to better understand any underlying mental health concerns.
"""

# Step 1: Generate questions for the entire text
questions = generate_questions_for_text(example_text)

# Step 2: Generate answers for the generated questions
qas = generate_answers(questions, example_text)

# Print the generated questions and answers
for qa in qas:
    print(f"Question: {qa['question']}")
    print(f"Answer: {qa['answer']}\n")
