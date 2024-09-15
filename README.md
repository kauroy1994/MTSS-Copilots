# Documentation

This is a **Question-Answer Knowledge Search System** that processes queries, retrieves top matched questions from a JSON dataset, and visualizes the results. The system displays extracted knowledge panels (subject-verb-object relationships) and provides a bar chart of the top matched questions with their similarity scores.

## Features:
- **Query Search**: Perform searches on the dataset of questions and answers.
- **Knowledge Panel**: Extract and display subject-verb-object triples from both the query and top matched answers.
- **Visualization**: Display a bar chart of the top matched questions and their similarity scores.
- **Suggested Queries**: Display a list of suggested queries from the dataset.

## Requirements

- Python 3.7+
- The following Python packages:
  - `gradio`
  - `spacy`
  - `sentence-transformers`
  - `torch`
  - `matplotlib`
  - `pillow`
  - `numpy`

## Setup

### 1. Clone the Repository

```bash
git clone <repository_url>.git
cd <respository_name>
```

### 2. Install Dependencies

Make sure to create and activate a Python virtual environment if necessary:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 3. Download the Spacy Model

You need to download the **Spacy model** for the project. You can do so by running:

```bash
python -m spacy download en_core_web_sm
```

### 4. Prepare the Dataset

Ensure you have a `JSON` file with the format:

```json
[
    {
        "question": "What is the role of administrators in MTSS?",
        "answer": "Administrators oversee the MTSS process..."
    },
    {
        "question": "How does MTSS help students?",
        "answer": "MTSS provides targeted interventions..."
    }
]
```

Place this JSON file in the project folder, and make sure to adjust the file path in the code accordingly.

### 5. Run the Application

Once you have set up everything, you can run the Gradio application with the following command:

```bash
python main.py
```

This will launch a local web server. Open the provided URL in your web browser to access the **Question-Answer Knowledge Search System**.

## Usage

- Enter a query in the search box and click "Search" to retrieve matched questions and answers.
- The system will display:
  - The **top matched question** and its corresponding **answer**.
  - A **knowledge panel** with subject-verb-object relationships extracted from both the query and the top matched answer.
  - A **bar chart** of the top matched questions with similarity scores.
  - A list of **suggested queries** extracted from the dataset.

## License

This project is licensed under the MIT License.
