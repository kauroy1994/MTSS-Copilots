# Knowledge Representation Comparison Gradio App

This Gradio application allows users to compare and contrast different knowledge representations of a given text, such as a school student's case file. It provides multiple approaches, including relation extraction (visualized as a graph), summarization, and question-answer generation. The application uses state-of-the-art NLP models from Hugging Face and SpaCy for dynamic and interactive visualization.

## Features

- **Relation Extraction (SpaCy-based)**: Extracts relations using SpaCy's dependency parsing and NER. Relations are visualized as a directed graph.
- **Relation Extraction (Hugging Face)**: Utilizes Hugging Face's `t5-base` model for dynamic relation extraction and visualizes the relations as a directed graph.
- **Summarization**: Generates a summary of the input text using the `facebook/bart-large-cnn` model from Hugging Face.
- **Question and Answer (QA)**: Produces a formatted table of questions and their answers based on the entities detected in the text using Hugging Face's `distilbert-base-cased-distilled-squad` model.

## Installation

Ensure you have Python installed on your system. Clone the repository or download the code files.

### Install Required Libraries

```bash
pip install gradio transformers torch spacy matplotlib networkx
```

### Download the SpaCy Model

```bash
python -m spacy download en_core_web_sm
```

## How to Run the Application

After installing the required libraries and the SpaCy model, run the application using the following command:

```bash
python knowledge_repr.py
```

## Usage

1. **Input Text**: Enter the case file or any text you want to analyze in the input box.
2. **Select Approach**: Choose from the dropdown menu to select one of the following knowledge representation approaches:
   - **SpaCy-based Relation Extraction**: Uses SpaCy’s dependency parsing to extract and visualize relations.
   - **Hugging Face Relation Extraction**: Dynamically extracts relations using Hugging Face's `t5-base` model and visualizes them as a graph.
   - **Summarization**: Generates a concise summary of the input text.
   - **Question and Answer**: Produces formatted questions and answers based on the entities detected in the text.
3. **View Results**: Depending on the selected approach, you will see:
   - A **relation graph** for the relation extraction approaches.
   - A **plain text summary** for the summarization approach.
   - A **formatted HTML table** for the QA approach.

## File Structure

- `knowledge_repr.py`: The main code file for the Gradio application.
- `requirements.txt`: (Optional) A list of dependencies that can be used with `pip install -r requirements.txt`.

## Implementation Details

### 1. Relation Extraction

- **SpaCy-based Relation Extraction**: This approach uses SpaCy’s dependency parsing to identify subjects, verbs, and objects in sentences. The relationships are then visualized as a directed graph using `networkx` and `matplotlib`.
- **Hugging Face Relation Extraction**: This approach uses the `t5-base` model to dynamically extract relations between entities detected by a NER pipeline. The graph is visualized similarly using `networkx`.

### 2. Summarization

The summarization model (`facebook/bart-large-cnn`) is used to generate a concise and relevant summary of the input text.

### 3. Question and Answer Generation

The application uses Hugging Face's `distilbert-base-cased-distilled-squad` model to generate questions and corresponding answers based on detected entities (persons) in the text. The output is formatted as an HTML table for clarity.

## Known Issues

- Ensure that the image is saved in the current directory (`"."`) if you encounter any file path issues with graph visualization.
- The application dynamically creates and deletes temporary files for graph visualization, ensuring compatibility across different operating systems.

## Troubleshooting

- **Graph Visualization Not Displaying**: Ensure you have write permissions in the directory where the code is running. The graph is saved as `relation_graph.png` in the current directory.
- **Dependencies**: If you encounter issues with missing packages, verify that all the required libraries are installed and that the versions are compatible.

## Future Improvements

- Add support for more advanced models or custom models for relation extraction and QA tasks.
- Enhance the interface with more customization options for graph layouts and colors.
- Allow users to export the generated graphs or summaries directly from the application.

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Gradio**: For providing the interactive web-based interface.
- **Hugging Face**: For the pre-trained models used for NER, summarization, relation extraction, and QA.
- **SpaCy**: For the dependency parsing and NER capabilities used in the SpaCy-based relation extraction approach.

