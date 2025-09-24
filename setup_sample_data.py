#!/usr/bin/env python3
"""
Setup script to create sample data structure for testing
Run this to create mock bank data if you don't have real PDFs
"""

import os
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def create_sample_data():
    """Create sample bank data for testing"""
    
    # Create data directory structure
    banks = ["icici", "sbi", "hdfc"]
    
    for bank in banks:
        # Create bank directory
        bank_dir = Path("data") / bank
        bank_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample CSV
        if bank == "icici":
            transactions = pd.DataFrame([
                {"Date": "01/01/2024", "Description": "Opening Balance", "Debit": "", "Credit": "10000.00", "Balance": "10000.00"},
                {"Date": "02/01/2024", "Description": "ATM Withdrawal", "Debit": "500.00", "Credit": "", "Balance": "9500.00"},
                {"Date": "03/01/2024", "Description": "Online Transfer", "Debit": "1200.00", "Credit": "", "Balance": "8300.00"},
                {"Date": "05/01/2024", "Description": "Salary Credit", "Debit": "", "Credit": "50000.00", "Balance": "58300.00"},
                {"Date": "07/01/2024", "Description": "Bill Payment", "Debit": "2500.00", "Credit": "", "Balance": "55800.00"},
            ])
        elif bank == "sbi":
            transactions = pd.DataFrame([
                {"Date": "2024-01-01", "Narration": "BY TRANSFER", "Withdrawal": 0, "Deposit": 5000, "Balance": 5000},
                {"Date": "2024-01-03", "Narration": "TO ATM", "Withdrawal": 1000, "Deposit": 0, "Balance": 4000},
                {"Date": "2024-01-05", "Narration": "BY SALARY", "Withdrawal": 0, "Deposit": 45000, "Balance": 49000},
            ])
        else:  # hdfc
            transactions = pd.DataFrame([
                {"Date": "01/01/24", "Particulars": "BALANCE B/F", "Dr": None, "Cr": 15000.0, "Balance": 15000.0},
                {"Date": "02/01/24", "Particulars": "CASH WITHDRAWAL", "Dr": 2000.0, "Cr": None, "Balance": 13000.0},
                {"Date": "04/01/24", "Particulars": "NEFT CREDIT", "Dr": None, "Cr": 25000.0, "Balance": 38000.0},
            ])
        
        # Save CSV
        csv_path = bank_dir / f"{bank}_sample.csv"
        transactions.to_csv(csv_path, index=False)
        print(f"✓ Created {csv_path}")
        
        # Create sample PDF
        pdf_path = bank_dir / f"{bank}_sample.pdf"
        create_sample_pdf(pdf_path, bank, transactions)
        print(f"✓ Created {pdf_path}")


def create_sample_pdf(pdf_path: Path, bank: str, transactions: pd.DataFrame):
    """Create a sample bank statement PDF"""
    
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height - 1*inch, f"{bank.upper()} BANK STATEMENT")
    
    c.setFont("Helvetica", 10)
    c.drawString(1*inch, height - 1.3*inch, "Account No: XXXX1234")
    c.drawString(1*inch, height - 1.5*inch, "Statement Period: 01/01/2024 to 31/01/2024")
    
    # Table header
    y_position = height - 2*inch
    c.setFont("Helvetica-Bold", 10)
    
    # Draw column headers based on bank
    if bank == "icici":
        headers = ["Date", "Description", "Debit", "Credit", "Balance"]
        x_positions = [1*inch, 2*inch, 4*inch, 5*inch, 6*inch]
    elif bank == "sbi":
        headers = ["Date", "Narration", "Withdrawal", "Deposit", "Balance"]
        x_positions = [1*inch, 2*inch, 4*inch, 5*inch, 6*inch]
    else:  # hdfc
        headers = ["Date", "Particulars", "Dr", "Cr", "Balance"]
        x_positions = [1*inch, 2*inch, 4*inch, 5*inch, 6*inch]
    
    for header, x in zip(headers, x_positions):
        c.drawString(x, y_position, header)
    
    # Draw transactions
    c.setFont("Helvetica", 9)
    y_position -= 20
    
    for _, row in transactions.iterrows():
        for col, x in zip(transactions.columns, x_positions):
            value = str(row[col]) if pd.notna(row[col]) else ""
            c.drawString(x, y_position, value)
        y_position -= 15
        
        if y_position < 2*inch:
            c.showPage()
            y_position = height - 1*inch
    
    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(1*inch, 1*inch, "This is a computer generated statement")
    
    c.save()


if __name__ == "__main__":
    print("🔧 Setting up sample data structure...")
    
    # Create directories
    Path("custom_parser").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Create sample data
    create_sample_data()
    
    print("\n✅ Sample data setup complete!")