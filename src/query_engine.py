import pandas as pd
import os

class CricketQueryEngine:
    def __init__(self, data_dir="data/cricket"):
        self.data_dir = data_dir
        self.batting_data = {}
        self.bowling_data = {}
        self.fielding_data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all cricket data"""
        # Batting
        self.batting_data['odi'] = pd.read_csv(f"{self.data_dir}/Batting/ODI data.csv")
        self.batting_data['t20'] = pd.read_csv(f"{self.data_dir}/Batting/t20.csv")
        self.batting_data['test'] = pd.read_csv(f"{self.data_dir}/Batting/test.csv")
        
        # Bowling
        self.bowling_data['odi'] = pd.read_csv(f"{self.data_dir}/Bowling/Bowling_ODI.csv")
        self.bowling_data['t20'] = pd.read_csv(f"{self.data_dir}/Bowling/Bowling_t20.csv")
        self.bowling_data['test'] = pd.read_csv(f"{self.data_dir}/Bowling/Bowling_test.csv")
        
        # Fielding
        self.fielding_data['odi'] = pd.read_csv(f"{self.data_dir}/Fielding/Fielding_ODI.csv")
        self.fielding_data['t20'] = pd.read_csv(f"{self.data_dir}/Fielding/Fielding_t20.csv")
        self.fielding_data['test'] = pd.read_csv(f"{self.data_dir}/Fielding/Fielding_test.csv")
        
        # Clean data
        self._clean_data()
    
    def _clean_data(self):
        """Clean and convert data types"""
        for format_name, df in self.batting_data.items():
            df['Runs'] = pd.to_numeric(df['Runs'], errors='coerce').fillna(0)
            df['Mat'] = pd.to_numeric(df['Mat'], errors='coerce').fillna(0)
            df['Ave'] = pd.to_numeric(df['Ave'], errors='coerce').fillna(0)
            if 'SR' in df.columns:
                df['SR'] = pd.to_numeric(df['SR'], errors='coerce').fillna(0)
            df['100'] = pd.to_numeric(df['100'], errors='coerce').fillna(0)
            df['50'] = pd.to_numeric(df['50'], errors='coerce').fillna(0)
            if '4s' in df.columns:
                df['4s'] = pd.to_numeric(df['4s'], errors='coerce').fillna(0)
            if '6s' in df.columns:
                df['6s'] = pd.to_numeric(df['6s'], errors='coerce').fillna(0)
        
        for format_name, df in self.bowling_data.items():
            df['Wkts'] = pd.to_numeric(df['Wkts'], errors='coerce').fillna(0)
            df['Mat'] = pd.to_numeric(df['Mat'], errors='coerce').fillna(0)
            df['Ave'] = pd.to_numeric(df['Ave'], errors='coerce').fillna(0)
            df['Econ'] = pd.to_numeric(df['Econ'], errors='coerce').fillna(0)
            if 'SR' in df.columns:
                df['SR'] = pd.to_numeric(df['SR'], errors='coerce').fillna(0)
    
    def get_most_fours(self, format_type='odi'):
        """Get player with most fours"""
        format_type = format_type.lower()
        df = self.batting_data.get(format_type)
        
        if df is None or '4s' not in df.columns:
            return None
        
        top_player = df.nlargest(1, '4s').iloc[0]
        return {
            'player': top_player['Player'],
            'fours': int(top_player['4s']),
            'format': format_type.upper(),
            'matches': int(top_player['Mat']),
            'runs': int(top_player['Runs'])
        }
    
    def get_most_sixes(self, format_type='odi'):
        """Get player with most sixes"""
        format_type = format_type.lower()
        df = self.batting_data.get(format_type)
        
        if df is None or '6s' not in df.columns:
            return None
        
        top_player = df.nlargest(1, '6s').iloc[0]
        return {
            'player': top_player['Player'],
            'sixes': int(top_player['6s']),
            'format': format_type.upper(),
            'matches': int(top_player['Mat']),
            'runs': int(top_player['Runs'])
        }
    
    def get_most_matches_batting(self, format_type='odi'):
        """Get player with most matches (batting)"""
        format_type = format_type.lower()
        df = self.batting_data.get(format_type)
        
        if df is None:
            return None
        
        top_player = df.nlargest(1, 'Mat').iloc[0]
        return {
            'player': top_player['Player'],
            'matches': int(top_player['Mat']),
            'format': format_type.upper(),
            'runs': int(top_player['Runs']),
            'average': float(top_player['Ave'])
        }
    
    def get_most_wickets(self, format_type='odi'):
        """Get player with most wickets"""
        format_type = format_type.lower()
        df = self.bowling_data.get(format_type)
        
        if df is None:
            return None
        
        top_player = df.nlargest(1, 'Wkts').iloc[0]
        return {
            'player': top_player['Player'],
            'wickets': int(top_player['Wkts']),
            'format': format_type.upper(),
            'matches': int(top_player['Mat']),
            'average': float(top_player['Ave']),
            'economy': float(top_player['Econ'])
        }
    
    def get_most_runs(self, format_type='odi'):
        """Get player with most runs"""
        format_type = format_type.lower()
        df = self.batting_data.get(format_type)
        
        if df is None:
            return None
        
        top_player = df.nlargest(1, 'Runs').iloc[0]
        return {
            'player': top_player['Player'],
            'runs': int(top_player['Runs']),
            'format': format_type.upper(),
            'matches': int(top_player['Mat']),
            'average': float(top_player['Ave']),
            'centuries': int(top_player['100']),
            'fifties': int(top_player['50'])
        }
    
    def get_player_stats(self, player_name, format_type='odi', category='batting'):
        """Get specific player statistics"""
        format_type = format_type.lower()
        category = category.lower()
        
        if category == 'batting':
            df = self.batting_data.get(format_type)
        elif category == 'bowling':
            df = self.bowling_data.get(format_type)
        else:
            df = self.fielding_data.get(format_type)
        
        if df is None:
            return None
        
        # Search for player (case insensitive, partial match)
        player_data = df[df['Player'].str.contains(player_name, case=False, na=False)]
        
        if player_data.empty:
            return None
        
        return player_data.iloc[0].to_dict()
    
    def answer_query(self, query):
        """Intelligent query answering"""
        query_lower = query.lower()
        
        # Detect format
        if 'odi' in query_lower:
            format_type = 'odi'
        elif 't20' in query_lower or 't-20' in query_lower:
            format_type = 't20'
        elif 'test' in query_lower:
            format_type = 'test'
        else:
            # Try to detect from context
            if 'twenty' in query_lower or '20 over' in query_lower:
                format_type = 't20'
            elif 'one day' in query_lower or '50 over' in query_lower:
                format_type = 'odi'
            elif 'five day' in query_lower:
                format_type = 'test'
            else:
                format_type = None  # Will answer for all formats
        
        # Most fours
        if 'most four' in query_lower or 'most 4s' in query_lower or 'highest four' in query_lower:
            if format_type:
                result = self.get_most_fours(format_type)
                if result:
                    return f"🏏 **{result['player']}** has hit the most fours in {result['format']} cricket with **{result['fours']} fours** in {result['matches']} matches, scoring {result['runs']} runs."
            else:
                # Show for all formats with fours data
                answers = []
                for fmt in ['t20']:
                    result = self.get_most_fours(fmt)
                    if result:
                        answers.append(f"**{fmt.upper()}:** {result['player']} - {result['fours']} fours")
                if answers:
                    return "🏏 **Most Fours:**\n\n" + "\n\n".join(answers)
        
        # Most sixes
        if 'most six' in query_lower or 'most 6s' in query_lower or 'highest six' in query_lower:
            if format_type:
                result = self.get_most_sixes(format_type)
                if result:
                    return f"💥 **{result['player']}** has hit the most sixes in {result['format']} cricket with **{result['sixes']} sixes** in {result['matches']} matches, scoring {result['runs']} runs."
            else:
                answers = []
                for fmt in ['t20']:
                    result = self.get_most_sixes(fmt)
                    if result:
                        answers.append(f"**{fmt.upper()}:** {result['player']} - {result['sixes']} sixes")
                if answers:
                    return "💥 **Most Sixes:**\n\n" + "\n\n".join(answers)
        
        # Most matches
        if 'most match' in query_lower or 'most game' in query_lower or 'played most' in query_lower:
            if format_type:
                result = self.get_most_matches_batting(format_type)
                if result:
                    return f"🏆 **{result['player']}** has played the most {result['format']} matches with **{result['matches']} matches**, scoring {result['runs']} runs at an average of {result['average']:.2f}."
            else:
                answers = []
                for fmt in ['odi', 't20', 'test']:
                    result = self.get_most_matches_batting(fmt)
                    if result:
                        answers.append(f"**{fmt.upper()}:** {result['player']} - {result['matches']} matches")
                if answers:
                    return "🏆 **Most Matches Played:**\n\n" + "\n\n".join(answers)
        
        # Most wickets
        if 'most wicket' in query_lower or 'top bowler' in query_lower or 'highest wicket' in query_lower:
            if format_type:
                result = self.get_most_wickets(format_type)
                if result:
                    return f"🎳 **{result['player']}** has taken the most wickets in {result['format']} cricket with **{result['wickets']} wickets** in {result['matches']} matches, with an average of {result['average']:.2f} and economy of {result['economy']:.2f}."
            else:
                answers = []
                for fmt in ['odi', 't20', 'test']:
                    result = self.get_most_wickets(fmt)
                    if result:
                        answers.append(f"**{fmt.upper()}:** {result['player']} - {result['wickets']} wickets")
                if answers:
                    return "🎳 **Most Wickets:**\n\n" + "\n\n".join(answers)
        
        # Most runs
        if 'most run' in query_lower or 'highest run' in query_lower or 'top scorer' in query_lower or 'top run' in query_lower:
            if format_type:
                result = self.get_most_runs(format_type)
                if result:
                    return f"🏏 **{result['player']}** has scored the most runs in {result['format']} cricket with **{result['runs']} runs** in {result['matches']} matches, averaging {result['average']:.2f} with {result['centuries']} centuries and {result['fifties']} half-centuries."
            else:
                answers = []
                for fmt in ['odi', 't20', 'test']:
                    result = self.get_most_runs(fmt)
                    if result:
                        answers.append(f"**{fmt.upper()}:** {result['player']} - {result['runs']} runs")
                if answers:
                    return "🏏 **Most Runs Scored:**\n\n" + "\n\n".join(answers)
        
        # Most centuries
        if 'most centur' in query_lower or 'most 100' in query_lower or 'most hundred' in query_lower:
            if format_type:
                df = self.batting_data.get(format_type)
                if df is not None:
                    top = df.nlargest(1, '100').iloc[0]
                    return f"💯 **{top['Player']}** has scored the most centuries in {format_type.upper()} cricket with **{int(top['100'])} centuries** in {int(top['Mat'])} matches."
        
        return None

if __name__ == "__main__":
    engine = CricketQueryEngine()
    
    # Test queries
    test_queries = [
        "Who has the most fours in ODI?",
        "Most sixes in T20?",
        "Player with most matches in Test cricket?",
        "Who has most wickets in ODI?",
        "Top run scorer in T20?"
    ]
    
    print("Testing Cricket Query Engine\n" + "="*60)
    for query in test_queries:
        print(f"\nQ: {query}")
        answer = engine.answer_query(query)
        print(f"A: {answer}")
