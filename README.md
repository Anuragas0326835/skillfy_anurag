# Skillfy Anurag

## Project Overview
Skillfy Anurag is a Machine Learning experimentation and demonstration repository aimed at providing insights into various ML models and methods. This repository serves as a practical guide for developers and data scientists to experiment with and understand machine learning concepts.

## Repository Composition
The repository consists of:
- **96.8% Jupyter Notebooks**: These are interactive notebooks that contain code, visualizations, and explanations.
- **3.2% Python scripts**: Supporting scripts for model training, data processing, and evaluation.

## Project Directories

### MLflow Demo
This directory provides an experiment tracking and model management solution using MLflow v3.8.1. The notebook covers the environment setup and showcases model training capabilities with popular libraries including:
- **scikit-learn**: For machine learning algorithms
- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computing
- **matplotlib**: For data visualization
- Flask & FastAPI: For web service deployment
- SQLAlchemy: For database operations
- Docker: For containerization

### Streamlit ML Application
The `stream_ML_demo` directory contains an interactive Iris Species Prediction App that demonstrates real-time machine learning inference. Key features include:
- **Pre-trained Random Forest Model** (`random_forest_model.pkl`): Trained on the classic Iris dataset
- **Interactive UI**: Users can adjust sepal length, sepal width, petal length, and petal width using sliders
- **Real-time Predictions**: Instant iris species classification based on input parameters
- **User-friendly Interface**: Built with Streamlit for easy deployment and interaction

**Usage**: Run `streamlit run app.py` from the `stream_ML_demo` directory to launch the application.

### DVC Integration
This section sets up Data Version Control (DVC) for efficient tracking of datasets in the project:
- `.dvc` directory: Contains DVC configuration files
- `data.dvc`: Tracks dataset versions
- `.dvcignore`: Specifies files to ignore from version control
- Ensures reproducibility and easy access to data changes across team members

### DagHub Demo
A collaborative machine learning pipeline management feature that includes:
- **diabetes.csv** (23KB): A comprehensive diabetes dataset for model training and evaluation
- Integration with DagHub for seamless team collaborations on ML projects
- Version control and experiment tracking capabilities

### Pytest Directory
Dedicated to unit testing and quality assurance:
- Test suite setup for validating model predictions
- Code quality checks
- Validation through various testing scenarios

### Code for Image Data
A placeholder directory for future image processing capabilities and computer vision demonstrations.

## Tech Stack & Dependencies
Major dependencies used in this repository:
- **MLflow 3.8.1**: Experiment tracking and model management
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **Streamlit**: Web app framework
- **DVC**: Data version control
- **Pytest**: Testing framework
- **FastAPI & Flask**: Web frameworks
- **SQLAlchemy**: ORM for database operations
- **Docker**: Container management

## Usage Instructions

### Getting Started
1. Clone the repository: `git clone https://github.com/Anuragas0326835/skillfy_anurag.git`
2. Navigate to the project: `cd skillfy_anurag`
3. Create a virtual environment: `python -m venv .venv`
4. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`

### Running Each Component

**MLflow Demo:**
- Open and run `mlflow_demo/mlflow_demo.ipynb` in Jupyter Notebook
- MLflow UI will be available at `http://localhost:5000`

**Streamlit ML Application:**
```bash
cd stream_ML_demo
streamlit run app.py
```
Access the app at `http://localhost:8501`

**DVC Integration:**
- Initialize DVC: `dvc init`
- Add datasets: `dvc add data/your_dataset.csv`
- Push to remote: `dvc push`

**Running Tests:**
```bash
pytest
```

**DagHub Integration:**
- Link your DagHub account
- Sync projects for collaborative development

## Model Details

### Iris Species Prediction Model
- **Type**: Random Forest Classifier
- **Training Data**: Iris dataset (150 samples, 4 features)
- **Features**: Sepal length, sepal width, petal length, petal width
- **Output**: Iris species classification (Setosa, Versicolor, Virginica)
- **Model File**: `stream_ML_demo/random_forest_model.pkl`

## Project Structure
```
skillfy_anurag/
├── README.md
├── data.dvc
├── .dvc/
├── .dvcignore
├── .gitignore
├── mlflow_demo/
│   ├── mlflow_demo.ipynb
│   └── test_mlflow_demo.ipynb
├── stream_ML_demo/
│   ├── app.py
│   └── random_forest_model.pkl
├── dagshub_demo/
│   └── diabetes.csv
├── code_for_image_data/
│   └── code_for_image_data.txt
└── pytest/
```

## Contributing
Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License
This project is open source and available under the MIT License.

---
**Author**: Anurag
**Created**: 2026
**Last Updated**: 2026-04-06 06:03:04

For questions or support, please open an issue on GitHub or contact the repository maintainer.