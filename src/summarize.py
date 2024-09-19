# Import necessary libraries
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Function to summarize the case file text
def summarize_case(case_text: str, model_name: str = "t5-small", max_length: int = 200, min_length: int = 30) -> str:
    """
    Summarizes a student's case file using the T5 transformer model.

    Args:
        case_text (str): The text of the case file to summarize.
        model_name (str): The name of the pre-trained T5 model to use for summarization.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.

    Returns:
        str: The summarized text.
    """
    
    # Load the T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Preprocess the text
    input_ids = tokenizer.encode("summarize: " + case_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary (adjust max_length and min_length as needed)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Example usage
if __name__ == "__main__":
    # Example case file text (replace this with the actual case file content)
    case_file_text = """
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

    # Summarize the case file
    summary = summarize_case(case_file_text)
    
    # Print the summarized text
    print("Summarized Case File:\n", summary)
