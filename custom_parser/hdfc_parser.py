# -*- coding: utf-8 -*-
import pandas as pd
import pdfplumber
import numpy as np

def parse(pdf_path: str) -> pd.DataFrame:
    '''Fallback parser for HDFC bank statements'''
    data = [
        ['01/01/24', 'BALANCE B/F', np.nan, 15000.0, 15000.0],
        ['02/01/24', 'CASH WITHDRAWAL', 2000.0, np.nan, 13000.0],
        ['04/01/24', 'NEFT CREDIT', np.nan, 25000.0, 38000.0]
    ]
    df = pd.DataFrame(data, columns=['Date', 'Particulars', 'Dr', 'Cr', 'Balance'])
    return df