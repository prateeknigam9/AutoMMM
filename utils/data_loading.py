# utilities/data_loading.py
import os
import pandas as pd
import logging
from utils.colors import system_message, system_input


def data_loading(processed_config, llm_client):
    input(system_input("📍 Press Enter to start the process..."))

    file_path = input(system_input("📂 Enter the path to your Excel file: ")).strip()
    while not os.path.isfile(file_path):
        logging.warning("Invalid file path entered.")
        file_path = input(system_input("❌ Invalid path. Please re-enter Excel file path: ")).strip()

    logging.info(f"Loading Excel file from: {file_path}")
    excel_file = pd.ExcelFile(file_path)
    logging.info(f"Found sheets: {excel_file.sheet_names}")
    print((system_message("📄 sheet_names: "), excel_file.sheet_names))

    sheet_name = input(system_input("📄 Sheet: ")).strip()
    while sheet_name not in excel_file.sheet_names:
        logging.warning("Invalid sheet name entered.")
        sheet_name = input(system_input("❌ Invalid sheet name. Please re-enter: ")).strip()

    logging.info(f"Reading sheet: {sheet_name}")
    processed_config["master_data"] = pd.read_excel(file_path, sheet_name)

    print(system_message("\n📝 Please describe the context of the data (e.g., what is the data about, business context, etc.)"))
    data_context = input(system_input("🧠 [Context Setting] The data is about : "))

    logging.info("Sending data to LLM for summarization...")
    print(system_message("\n💬 Summarizing your input using Ollama..."))

    column_desc = llm_client.chat(
        model=processed_config["model"],
        messages=[{
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
                """.strip()
        }],
    )

    column_desc = column_desc["message"]["content"]
    # column_desc = ""
    logging.info("LLM response received and printed below.")
    print(system_message("\n📌 Column Description:"))
    print(column_desc)

    os.makedirs("memory", exist_ok=True)
    with open("memory/column_desc.txt", "w", encoding="utf-8") as f:
        f.write("column description:\n" + str(column_desc))

    logging.info("Column description saved to memory/column_desc.txt")
    print(system_message("\n💾 Summary saved to memory"))

    while True:
        user_input = input(
            system_input("\nAfter editing the file in memory --> column description text file\n Type 'continue' to proceed: ")
            
        )
        if user_input.lower() == "continue":
            logging.info("User confirmed continuation after editing.")
            print(system_message("Continuing..."))
            break
        else:
            logging.info("Waiting for user to finish editing column_desc.txt")
            print(system_message("Waiting for file update..."))
