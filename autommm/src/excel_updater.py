import win32com.client as win32
import sys
import os

# ---- TEMP FIX FOR MODULE IMPORTS ----
# Adds project root (3 levels up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# https://learndataanalysis.org/automate-excel-pivot-table-with-python/
# https://xlsxwriter.readthedocs.io/contents.html

def clear_pts(ws):
    for pt in ws.PivotTables():
        pt.TableRange2.Clear()

def insert_pt_fields(pt, row_fields, col_fields, data_fields):
    field_rows = {}
    field_cols = {}
    field_values = {}

    for idx, fld in enumerate(row_fields, start=1):
        field_rows[fld] = pt.PivotFields(fld)
        field_rows[fld].Orientation = 1  # xlRowField
        field_rows[fld].Position = idx
        field_rows[fld].Subtotals = tuple(False for _ in range(12))

    for idx, fld in enumerate(col_fields, start=1):
        field_cols[fld] = pt.PivotFields(fld)
        field_cols[fld].Orientation = 2  # xlColumnField
        field_cols[fld].Position = idx
        field_cols[fld].Subtotals = tuple(False for _ in range(12))

    for src, field_name, func, numfmt in data_fields:
        field_values[field_name] = pt.PivotFields(src)
        field_values[field_name].Orientation = 4  # xlDataField
        field_values[field_name].Function = func
        field_values[field_name].NumberFormat = numfmt

def create_pivot_table(data_sheet, report_sheet, report_sheet_ref:str, title: str, 
                       row_fields: list, col_fields: list, data_fields: list[tuple],
                       grand_total: bool = False, row_total: bool = False,
                       report_axis_layout: int = 1, 
                       table_style: str = "PivotStyleMedium9", wb=None
                       ):

    clear_pts(report_sheet)

    pt_cache = wb.PivotCaches().Create(1, data_sheet.Range("A1").CurrentRegion)
    pt = pt_cache.CreatePivotTable(report_sheet.Range(report_sheet_ref), title)

    pt.ColumnGrand = grand_total
    pt.RowGrand = row_total
    pt.RowAxisLayout(report_axis_layout)
    pt.TableStyle2 = table_style

    insert_pt_fields(pt, row_fields, col_fields, data_fields)

    pt.RepeatAllLabels(2)  # Excel constant = 2 :contentReference[oaicite:2]{index=2}

    return pt

def main():
    xlApp = win32.Dispatch('Excel.Application')
    xlApp.Visible = True

    wb_path = r"C:\Users\nigam\OneDrive\Documents\university_classes\AutoMMM\autommm\data\automated.xlsx"
    wb = xlApp.Workbooks.Open(wb_path)

    ws_data = wb.Worksheets("data")
    ws_report = wb.Worksheets("report")

    create_pivot_table(
        data_sheet=ws_data,
        report_sheet=ws_report,
        report_sheet_ref="B3",
        title="myreportsummary",
        row_fields=['date','sku'],
        col_fields=[],
        data_fields=[
            ("sales", "Total Sales", -4157, "$#,##0"),
            ("units", "Units Count", -4112, "#,##0")
        ],
        grand_total=False,
        row_total=False,
        report_axis_layout=1,
        table_style="PivotStyleMedium9",
        wb=wb
    )

    # Save workbook if changes were made
    # wb.Save()
    # # Close workbook and quit Excel
    # wb.Close(SaveChanges=False)
    # xlApp.Quit()

if __name__ == "__main__":
    main()
