from utils import *

base_url = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
season = "1415"

df = get_match_data(base_url.format(season))
print(df.head())
