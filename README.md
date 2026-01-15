# MLM-DE-MOO Project

Multi-objective optimization project using Minimal Learning Machine for explainability.

## Setup

### Virtual Environment

A Python virtual environment has been created in the `venv` directory with all required dependencies.

### Activation

To activate the virtual environment:

```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### Deactivation

To deactivate the virtual environment:

```bash
deactivate
```

## Dependencies

The following packages are installed:
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- tqdm >= 4.62.0
- pymoo >= 0.6.0
- matplotlib >= 3.4.0

## Running Tests

After activating the virtual environment, you can run the tests:

```bash
python tests/test_DTLZ1_2.py
```

## Project Structure

- `mlm_explainability.py` - Main MLM regressor and explainer classes
- `tests/` - Test files
- `venv/` - Virtual environment (not tracked in git)
- `requirements.txt` - Python dependencies
