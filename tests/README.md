## Test Notes

#### Run the tests
Tests are run using the pytest command in the root of the project. Described as below:

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