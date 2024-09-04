# Documentation

## Setup
```
pip install -r requirements.txt
```

## Navigate to source folder
```
cd src
```

## To run a query without role-based alignment, in the src/main.py file line containing "Copilot.run", replace with
```python
response = Copilot.run(config_json,test_query = <query>, role=None)
```

## To run a query with role-based alignment, in the src/main.py file line containing "Copilot.run", replace with
```python
response = Copilot.run(config_json,test_query = <query>, role=<role>)
```
Available roles are:
```python
['admin','clinical_staff','school_counselor','school_psych','teachers']
```
