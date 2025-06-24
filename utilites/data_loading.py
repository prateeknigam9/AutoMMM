import os
import pandas as pd

def data_loading(processed_config, llm_client):
    input("📍 Press Enter to start the process...")

    file_path = input("📂 Enter the path to your Excel file: ").strip()
    while not os.path.isfile(file_path):
        file_path = input("❌ Invalid path. Please re-enter Excel file path: ").strip()

    excel_file = pd.ExcelFile(file_path)
    print("sheet_names: ", excel_file.sheet_names)

    sheet_name = input("📄 Sheet: ").strip()
    while sheet_name not in excel_file.sheet_names:
        sheet_name = input("❌ Invalid sheet name. Please re-enter: ").strip()

    processed_config["master_data"] = pd.read_excel(file_path, sheet_name)

    print("\n📝 Please describe the context of the data (e.g., what is the data about, business context, etc.)")
    data_context = input("🧠 [Context Setting] The data is about : ")

    print("\n💬 Summarizing your input using Ollama...")

    column_desc = llm_client.chat(
        model=processed_config["model"],
        messages=[
            {
                "role": "user",
                "content": f"""
                    List the Market Mix Modeling dataset columns as a JSON dictionary, 
                    with each column followed by a brief one-line description of its meaning in marketing mix context.
                    Use simple language.

                    Example:
                    {{
                        "date": "Observation date",
                        "product": "Product descriptions",
                        "sales": "Revenue in euros"
                    }}

                    Data sample (first 3 rows): {processed_config['master_data'].head(3).to_dict(orient='records')}

                    Context: {data_context}

                    Please respond only with a valid JSON dictionary.
                    """.strip(),
            }
        ],
    )

    column_desc = column_desc["message"]["content"]

    print("\n📌 Column Description:")
    print(column_desc)
    os.makedirs("memory", exist_ok=True)
    with open("memory/column_desc.txt", "w", encoding="utf-8") as f:
        f.write("column description:\n" + str(column_desc))

    print("\n💾 Summary saved to memory")

    while True:
        user_input = input(
            "\nAfter editing the file in memory --> column description text file\n"
            "Type 'continue' to proceed: "
        )
        if user_input.lower() == "continue":
            print("Continuing...")
            break
        else:
            print("Waiting for file update...")
