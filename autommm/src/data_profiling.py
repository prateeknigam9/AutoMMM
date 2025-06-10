# profile_report.py
import sys
import subprocess
import pandas as pd

def install_packages():
    try:
        import ydata_profiling
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ydata-profiling"])

def main():
    if len(sys.argv) != 3:
        print("Usage: python profile_report.py <input_file> <output_html>")
        sys.exit(1)

    install_packages()

    from ydata_profiling import ProfileReport

    input_file = sys.argv[1]
    output_html = sys.argv[2]

    if input_file.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    profile = ProfileReport(df, title="Data Profiling Report")
    profile.to_file(output_html)
    print(f"Profiling report saved to {output_html}")

if __name__ == "__main__":
    main()
