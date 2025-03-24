from utils.data_loader import load_csv

def test_load_csv():
    df = load_csv("sample_payroll.csv")
    assert df is not None, "CSV file should load successfully"
