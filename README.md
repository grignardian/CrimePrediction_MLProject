# Crime Prediction System

This is a college ML project that predicts total cognizable IPC crimes in India districts using census data. 

We merged district-level data from the 2011 Census of India with the 2011 National Crime Records Bureau (NCRB) dataset to see how demographic variables (like population, growth, sex ratio, and literacy) correlate with crime rates.

## How it Works
1. **Data Cleaning & Matching:** We load `census.csv` and `NCRB_2011_Table_1.14.csv` and match the district names. E.g., handling spelling differences across both datasets. Out of 642 districts, we successfully matched 618.
2. **Model Training:** We train two models on an 80/20 train-test split:
   - **Random Forest Regressor** (which has an R² of ~0.56)
   - **Multiple Linear Regression** (which has an R² of ~0.43)
3. **Interactive UI:** Built a Streamlit interface where you can input custom demographics to predict crime volumes and compare model metrics.

## Datasets Used
- `census.csv`: Contains population, growth rates, sex ratio, and literacy rates.
- `NCRB_2011_Table_1.14.csv`: Contains IPC crimes data for 2011.

## Getting Started

### Prerequisites
You need Python installed with the following packages:
```bash
pip install pandas scikit-learn matplotlib streamlit
```

### Running the App
To start the Streamlit UI, run:
```bash
streamlit run project.py
```
This will open the dashboard in your web browser.
