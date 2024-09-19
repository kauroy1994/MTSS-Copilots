# MTSS Case Processing using Natural Language Processing Tools

## 💻 Setup
```bash
pip install -r requirements.txt
```

## 🔧 Downloading spacy english model
```bash
python -m spacy download en_core_web_sm
```

## ▶️ Running the code
```
python src/<file_name>.py
```

## Change the variables ```case_file_text``` or ```example_text``` depending on the .py file as follows (example shows ```case_file_text```):
```python
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
```
