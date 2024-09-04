# Documentation

## Setup
```
pip install -r requirements.txt
```

## Navigate to source folder
```
cd src
```

## Set configuration in config.json file
Example configuration file
```json
{"GRADIO": "True", 
"query": "What is MTSS?",
"role":"admin",
"LLM_config":"None", 
"data_folder_path": "/assets/json_files"
}
```

## Execute QA Copilot
```python
python main.py
```
