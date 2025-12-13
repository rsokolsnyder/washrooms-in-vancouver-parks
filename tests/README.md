## Test Notes

#### Run the tests
Tests are run using the pytest command in the root of the project. 
Use terminal and run the commands listed below:

- download test:
```
pytest -q tests/test_download.py
```

- validation test:
```
pytest -q tests/test_validate_target_distribution.py
```

- fitting test:
```
pytest -q tests/test_fit_workflow.py
```

- evaluation test:
```
pytest -q tests/test_evaluate_workflow.py
```
