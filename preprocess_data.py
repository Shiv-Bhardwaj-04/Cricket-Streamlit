import pandas as pd
import numpy as np
import os
from pathlib import Path

class CricketDataProcessor:
    def __init__(self, data_dir="data/cricket"):
        self.data_dir = data_dir
        self.batting_data = {}
        self.bowling_data = {}
        self.fielding_data = {}
        
    def load_all_data(self):
        """Load all cricket data from CSV files"""
        print("Loading cricket data...")
        
        # Load Batting data
        self.batting_data['odi'] = pd.read_csv(f"{self.data_dir}/Batting/ODI data.csv")
        self.batting_data['t20'] = pd.read_csv(f"{self.data_dir}/Batting/t20.csv")
        self.batting_data['test'] = pd.read_csv(f"{self.data_dir}/Batting/test.csv")
        
        # Load Bowling data
        self.bowling_data['odi'] = pd.read_csv(f"{self.data_dir}/Bowling/Bowling_ODI.csv")
        self.bowling_data['t20'] = pd.read_csv(f"{self.data_dir}/Bowling/Bowling_t20.csv")
        self.bowling_data['test'] = pd.read_csv(f"{self.data_dir}/Bowling/Bowling_test.csv")
        
        # Load Fielding data
        self.fielding_data['odi'] = pd.read_csv(f"{self.data_dir}/Fielding/Fielding_ODI.csv")
        self.fielding_data['t20'] = pd.read_csv(f"{self.data_dir}/Fielding/Fielding_t20.csv")
        self.fielding_data['test'] = pd.read_csv(f"{self.data_dir}/Fielding/Fielding_test.csv")
        
        print("[OK] Data loaded successfully")
        self.print_data_summary()
        
    def print_data_summary(self):
        """Print summary of loaded data"""
        print("\n" + "="*60)
        print("CRICKET DATA SUMMARY")
        print("="*60)
        
        for category, data_dict in [("Batting", self.batting_data), 
                                     ("Bowling", self.bowling_data), 
                                     ("Fielding", self.fielding_data)]:
            print(f"\n{category}:")
            for format_name, df in data_dict.items():
                print(f"  {format_name.upper()}: {len(df)} players, {len(df.columns)} columns")
        
        total_records = sum(len(df) for data in [self.batting_data, self.bowling_data, self.fielding_data] 
                           for df in data.values())
        print(f"\nTotal Records: {total_records}")
        print("="*60 + "\n")
    
    def clean_data(self):
        """Clean and preprocess all datasets"""
        print("Cleaning data...")
        
        for category, data_dict in [("batting", self.batting_data), 
                                     ("bowling", self.bowling_data), 
                                     ("fielding", self.fielding_data)]:
            for format_name, df in data_dict.items():
                # Remove unnamed columns
                df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True, errors='ignore')
                
                # Clean player names
                if 'Player' in df.columns:
                    df['Player'] = df['Player'].str.strip()
                
                # Handle missing values
                df.fillna(0, inplace=True)
                
                # Add format and category labels
                df['Format'] = format_name.upper()
                df['Category'] = category.capitalize()
        
        print("[OK] Data cleaned successfully")
    
    def create_unified_dataset(self):
        """Create a unified dataset for training"""
        print("Creating unified dataset...")
        
        all_data = []
        
        # Combine all batting data
        for format_name, df in self.batting_data.items():
            df_copy = df.copy()
            df_copy['Type'] = 'Batting'
            all_data.append(df_copy)
        
        # Combine all bowling data
        for format_name, df in self.bowling_data.items():
            df_copy = df.copy()
            df_copy['Type'] = 'Bowling'
            all_data.append(df_copy)
        
        # Combine all fielding data
        for format_name, df in self.fielding_data.items():
            df_copy = df.copy()
            df_copy['Type'] = 'Fielding'
            all_data.append(df_copy)
        
        # Save unified dataset
        os.makedirs("processed_data", exist_ok=True)
        
        for i, df in enumerate(all_data):
            df.to_csv(f"processed_data/dataset_{i}.csv", index=False)
        
        print(f"[OK] Created {len(all_data)} processed datasets")
        return all_data
    
    def generate_text_corpus(self):
        """Generate text corpus for embedding training"""
        print("Generating text corpus...")
        
        corpus = []
        
        # Generate sentences from batting data
        for format_name, df in self.batting_data.items():
            for _, row in df.iterrows():
                player = row.get('Player', 'Unknown')
                runs = row.get('Runs', 0)
                avg = row.get('Ave', 0)
                sr = row.get('SR', 0)
                hundreds = row.get('100', 0)
                fifties = row.get('50', 0)
                
                corpus.append(f"{player} scored {runs} runs in {format_name.upper()} cricket with an average of {avg} and strike rate of {sr}. He has {hundreds} centuries and {fifties} half-centuries.")
        
        # Generate sentences from bowling data
        for format_name, df in self.bowling_data.items():
            for _, row in df.iterrows():
                player = row.get('Player', 'Unknown')
                wickets = row.get('Wkts', 0)
                avg = row.get('Ave', 0)
                econ = row.get('Econ', 0)
                
                corpus.append(f"{player} took {wickets} wickets in {format_name.upper()} cricket with a bowling average of {avg} and economy rate of {econ}.")
        
        # Save corpus
        with open("processed_data/cricket_corpus.txt", "w", encoding="utf-8") as f:
            for sentence in corpus:
                f.write(sentence + "\n")
        
        print(f"[OK] Generated corpus with {len(corpus)} sentences")
        return corpus
    
    def analyze_data_statistics(self):
        """Analyze and print data statistics"""
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        
        # Convert numeric columns
        for df in self.batting_data.values():
            if 'Runs' in df.columns:
                df['Runs'] = pd.to_numeric(df['Runs'], errors='coerce').fillna(0)
            if 'Ave' in df.columns:
                df['Ave'] = pd.to_numeric(df['Ave'], errors='coerce').fillna(0)
            if 'SR' in df.columns:
                df['SR'] = pd.to_numeric(df['SR'], errors='coerce').fillna(0)
        
        for df in self.bowling_data.values():
            if 'Wkts' in df.columns:
                df['Wkts'] = pd.to_numeric(df['Wkts'], errors='coerce').fillna(0)
            if 'Ave' in df.columns:
                df['Ave'] = pd.to_numeric(df['Ave'], errors='coerce').fillna(0)
            if 'Econ' in df.columns:
                df['Econ'] = pd.to_numeric(df['Econ'], errors='coerce').fillna(0)
        
        # Batting statistics
        print("\nTop 5 Run Scorers (ODI):")
        odi_batting = self.batting_data['odi'].nlargest(5, 'Runs')[['Player', 'Runs', 'Ave', 'SR']]
        print(odi_batting.to_string(index=False))
        
        # Bowling statistics
        print("\nTop 5 Wicket Takers (ODI):")
        odi_bowling = self.bowling_data['odi'].nlargest(5, 'Wkts')[['Player', 'Wkts', 'Ave', 'Econ']]
        print(odi_bowling.to_string(index=False))
        
        print("="*60 + "\n")

def main():
    processor = CricketDataProcessor()
    
    # Step 1: Load data
    processor.load_all_data()
    
    # Step 2: Clean data
    processor.clean_data()
    
    # Step 3: Analyze statistics
    processor.analyze_data_statistics()
    
    # Step 4: Create unified dataset
    processor.create_unified_dataset()
    
    # Step 5: Generate text corpus
    processor.generate_text_corpus()
    
    print("\n[SUCCESS] Data preprocessing completed successfully!")
    print("Next steps:")
    print("1. Run model training: python train_models.py")
    print("2. Evaluate models: python evaluate_models.py")
    print("3. Deploy best model: python deploy.py")

if __name__ == "__main__":
    main()
