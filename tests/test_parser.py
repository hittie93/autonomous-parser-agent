"""
Test suite for validating generated parsers
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import importlib.util


class TestParserValidation:
    """Test suite for bank parser validation"""
    
    @pytest.fixture
    def bank_data(self, request):
        """Load bank test data"""
        bank_name = request.param
        base_path = Path("data") / bank_name
        
        return {
            "bank": bank_name,
            "pdf_path": base_path / f"{bank_name}_sample.pdf",
            "csv_path": base_path / f"{bank_name}_sample.csv",
            "parser_path": Path("custom_parser") / f"{bank_name}_parser.py"
        }
    
    @pytest.mark.parametrize("bank_data", ["icici"], indirect=True)
    def test_parser_exists(self, bank_data):
        """Test that parser file was generated"""
        assert bank_data["parser_path"].exists(), \
            f"Parser not found: {bank_data['parser_path']}"
    
    @pytest.mark.parametrize("bank_data", ["icici"], indirect=True)
    def test_parser_contract(self, bank_data):
        """Test that parser has correct function signature"""
        # Load parser module
        spec = importlib.util.spec_from_file_location(
            f"{bank_data['bank']}_parser",
            bank_data["parser_path"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check parse function exists
        assert hasattr(module, "parse"), "Parser must have 'parse' function"
        
        # Check function signature
        import inspect
        sig = inspect.signature(module.parse)
        assert "pdf_path" in sig.parameters, \
            "parse() must accept 'pdf_path' parameter"
    
    @pytest.mark.parametrize("bank_data", ["icici"], indirect=True)
    def test_parser_output_schema(self, bank_data):
        """Test that parser output matches expected schema"""
        # Load expected data
        expected_df = pd.read_csv(bank_data["csv_path"])
        
        # Import and run parser
        spec = importlib.util.spec_from_file_location(
            f"{bank_data['bank']}_parser",
            bank_data["parser_path"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Parse PDF
        result_df = module.parse(str(bank_data["pdf_path"]))
        
        # Validate schema
        assert isinstance(result_df, pd.DataFrame), \
            "Parser must return pandas DataFrame"
        assert list(result_df.columns) == list(expected_df.columns), \
            f"Column mismatch: {list(result_df.columns)} != {list(expected_df.columns)}"
        assert result_df.shape == expected_df.shape, \
            f"Shape mismatch: {result_df.shape} != {expected_df.shape}"
    
    @pytest.mark.parametrize("bank_data", ["icici"], indirect=True)
    def test_parser_exact_match(self, bank_data):
        """Test that parser output exactly matches expected CSV"""
        # Load expected data
        expected_df = pd.read_csv(bank_data["csv_path"])
        
        # Import and run parser
        spec = importlib.util.spec_from_file_location(
            f"{bank_data['bank']}_parser",
            bank_data["parser_path"]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Parse PDF
        result_df = module.parse(str(bank_data["pdf_path"]))
        
        # Exact match validation
        pd.testing.assert_frame_equal(
            result_df, 
            expected_df,
            check_dtype=False,  # Allow minor dtype differences
            check_exact=False,   # Allow minor float differences
        )


def test_agent_cli():
    """Test agent CLI execution"""
    import subprocess
    
    # Run agent
    result = subprocess.run(
        ["python", "agent.py", "--target", "icici"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    # Check execution
    assert result.returncode == 0, f"Agent failed: {result.stderr}"
    assert "Parser validated successfully" in result.stdout