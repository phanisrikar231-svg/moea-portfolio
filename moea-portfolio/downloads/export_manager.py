import pandas as pd
from io import BytesIO
from fpdf import FPDF

def export_csv(pareto_df):
    return pareto_df.to_csv(index=False).encode("utf-8")

def export_excel(pareto_df, metrics_df, allocation_dict):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pareto_df.to_excel(w, sheet_name="Pareto Front", index=False)
        metrics_df.to_excel(w, sheet_name="Metrics Comparison", index=False)
        pd.DataFrame(list(allocation_dict.items()),
                     columns=["Stock", "Weight"]
                     ).to_excel(w, sheet_name="Allocation", index=False)
    return buf.getvalue()

def export_pdf(user_name, metrics_dict, allocation_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "MOEA Portfolio Optimization Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Prepared for: {user_name}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Performance Metrics", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for k, v in metrics_dict.items():
        pdf.cell(0, 8, f"  {k}: {v}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Portfolio Allocation", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for stock, wt in allocation_dict.items():
        pdf.cell(0, 8, f"  {stock}: {round(float(wt)*100, 2)}%", ln=True)
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()
