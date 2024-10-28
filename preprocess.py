import logging
from pathlib import Path
from typing import Union, Optional
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """A class to handle data preprocessing operations for different file formats."""
    
    @staticmethod
    def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard preprocessing steps to a DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        try:
            # Convert text columns to lowercase for standardization
            df = df.map(lambda x: x.lower() if isinstance(x, str) else x)
            
            # Drop columns that are fully null
            df = df.dropna(axis=1, how='all')
            
            # Fill remaining NaN values with empty strings
            df = df.fillna('')
            
            # Remove duplicate rows
            df = df.drop_duplicates()
            
            return df
            
        except Exception as e:
            logger.error(f"Error during DataFrame preprocessing: {str(e)}")
            raise

    @classmethod
    def preprocess_msd(cls, 
                      file_path: Union[str, Path], 
                      output_path: Union[str, Path],
                      sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
        """
        Preprocesses an MSD Excel file and saves the result.
        
        Args:
            file_path: Path to the Excel file
            output_path: Directory path for the output file
            sheet_name: Sheet name or index to load (default: 0)
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            PermissionError: If output directory is not writable
        """
        try:
            # Convert to Path objects
            file_path = Path(file_path)
            output_path = Path(output_path)
            
            # Validate input file
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")
            
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing MSD file: {file_path}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Apply preprocessing
            df = cls._preprocess_dataframe(df)
            
            # Save processed file
            output_file = output_path / "msd_processed.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved processed file to: {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing MSD file: {str(e)}")
            raise

    @classmethod
    def preprocess_cbip(cls,
                       input_dir: Union[str, Path],
                       output_dir: Union[str, Path]) -> None:
        """
        Preprocesses all CSV files in the CBIP directory.
        
        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory for output files
            
        Raises:
            FileNotFoundError: If input directory doesn't exist
            PermissionError: If output directory is not writable
        """
        try:
            # Convert to Path objects
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            
            # Validate input directory
            if not input_dir.exists():
                raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all CSV files
            csv_files = list(input_dir.rglob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in: {input_dir}")
                return
            
            for file_path in csv_files:
                try:
                    logger.info(f"Processing CBIP file: {file_path}")
                    
                    # Read CSV file
                    df = pd.read_csv(
                        file_path,
                        delimiter=';',
                        quotechar='"',
                        skip_blank_lines=True
                    )
                    
                    # Apply preprocessing
                    df = cls._preprocess_dataframe(df)
                    
                    # Save processed file
                    output_file = output_dir / file_path.name
                    df.to_csv(output_file, index=False)
                    logger.info(f"Saved processed file to: {output_file}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing CBIP directory: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        import os
        import argparse
        from pathlib import Path
        
        # Create processed_data directory in current working directory
        output_base = Path.cwd() / "processed_data"
        msd_output = output_base / "msd"
        cbip_output = output_base / "cbip"
        
        parser = argparse.ArgumentParser(description='Process MSD and CBIP data files.')
        parser.add_argument('--msd-input', required=True, help='Path to MSD Excel file')
        parser.add_argument('--cbip-input', required=True, help='Input directory containing CBIP CSV files')
        
        args = parser.parse_args()
        
        preprocessor = DataPreprocessor()
        
        # Process MSD file
        preprocessor.preprocess_msd(
            args.msd_input,
            msd_output
        )
        
        # Process CBIP directory
        preprocessor.preprocess_cbip(
            args.cbip_input,
            cbip_output
        )
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()