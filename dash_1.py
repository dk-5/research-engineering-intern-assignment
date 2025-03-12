import streamlit as st
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from urllib.parse import urlparse
from wordcloud import WordCloud
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from google import genai
import os 


nltk.download('stopwords')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def generate_summary_with_gemini(prompt):
  
    api_key = ''
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=[prompt],
    )
    print(response)
    return response.text


def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if "data" in record:
                data.append(record["data"])
    df = pd.json_normalize(data)
    if 'created_utc' in df.columns:
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    return df


def extract_keywords(data):
    stop_words = set(stopwords.words('english'))
    text_data = []
    if 'title' in data.columns:
        text_data += data['title'].dropna().tolist()
    if 'selftext' in data.columns:
        text_data += data['selftext'].dropna().tolist()
    words = []
    translator = str.maketrans("", "", string.punctuation)
    for text in text_data:
        cleaned_text = text.translate(translator)
        for word in cleaned_text.split():
            word = word.lower().strip()
            if len(word) > 3 and word not in stop_words:
                words.append(word)
    return Counter(words)


def extract_time_series(data, query):
    if 'created_utc' not in data.columns:
        return pd.Series()
    df = data.copy()
    if query:
        df = df[
            (df['title'].str.contains(query, case=False, na=False)) |
            (df['selftext'].str.contains(query, case=False, na=False))
        ]
    if df.empty:
        return pd.Series()
    df = df.set_index('created_utc')
    return df.resample('D').size()


def top_authors(data, n=10):
    if 'author' not in data.columns:
        return []
    return Counter(data['author'].dropna()).most_common(n)

def top_subreddits(data, n=10):
    if 'subreddit' not in data.columns:
        return []
    return Counter(data['subreddit'].dropna()).most_common(n)

def top_shared_domains(data, n=10):
    if 'url' not in data.columns:
        return []
    domains = []
    for url in data['url'].dropna():
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain:
            domains.append(domain)
    return Counter(domains).most_common(n)


def create_network(data):
    if 'author' not in data.columns or 'subreddit' not in data.columns:
        return nx.Graph()
    G = nx.Graph()
    for _, row in data.iterrows():
        author = row.get('author')
        subreddit = row.get('subreddit')
        if author and subreddit:
            G.add_node(author, type='author')
            G.add_node(subreddit, type='subreddit')
            G.add_edge(author, subreddit)
    return G

def detect_communities(G):
    if G.number_of_nodes() == 0:
        return {}
    communities = nx.community.greedy_modularity_communities(G)
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx
    return community_map


def post_activity(data):
    if 'created_utc' not in data.columns:
        return None, None
    df = data.copy().dropna(subset=['created_utc'])
    df['day_of_week'] = df['created_utc'].dt.day_name()
    df['hour'] = df['created_utc'].dt.hour
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(days_order)
    hour_counts = df['hour'].value_counts().sort_index()
    return day_counts, hour_counts


def generate_wordcloud(counter):
    if not counter:
        return None
    return WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(counter)


political_terms = {
    "politics", "government", "election", "vote", "trump", "biden", "democrat",
    "republican", "liberal", "conservative", "policy", "senate", "congress",
    "president", "political", "campaign", "scandal", "impeachment", "minister"
}

def filter_political_posts(data, terms):
    if 'title' in data.columns and 'selftext' in data.columns:
        title_match = data['title'].fillna("").str.lower().apply(lambda x: any(term in x for term in terms))
        selftext_match = data['selftext'].fillna("").str.lower().apply(lambda x: any(term in x for term in terms))
        mask = title_match | selftext_match
        return data[mask]
    return pd.DataFrame()

def political_sentiment_analysis(data):
    sentiments = []
    for _, row in data.iterrows():
        text = ""
        if pd.notnull(row.get('title')):
            text += row['title'] + " "
        if pd.notnull(row.get('selftext')):
            text += row['selftext']
        if text:
            score = sia.polarity_scores(text)
            sentiments.append(score)
    if sentiments:
        return pd.DataFrame(sentiments)
    else:
        return pd.DataFrame()


def generate_narrative_summary(total_posts, min_date, max_date, overall_keywords, political_data, political_keywords, sentiment_df):
    overall_top = overall_keywords.most_common(1)[0] if overall_keywords else ("N/A", 0)
    prompt = f"""Summarize the below given insights:
Our Reddit dataset contains {total_posts} posts collected between {min_date.date()} and {max_date.date()}.
The discussions are heavily focused on political topics. Overall, the top keyword is "{overall_top[0]}" with {overall_top[1]} mentions.
Within the political subset, keywords such as "election", "government", and notably "trump" appear frequently.
"Trump" is the most used word, likely due to the persistent public focus on his controversial policies and the polarized debates.
Sentiment analysis on political posts shows an average compound sentiment score of {sentiment_df['compound'].mean():.2f}, suggesting emotionally charged discussions.
Network analysis reveals distinct clusters of influential authors and key political subreddits.
Please provide a concise summary that explains these insights and discusses why "trump" stands out.
"""
    return generate_summary_with_gemini(prompt)


def generate_subreddit_summary(subreddit, data):
    
    sub_data = data[data['subreddit'] == subreddit]
    if sub_data.empty:
        return "No posts found for the selected subreddit."
    
    
    if 'score' in sub_data.columns:
        top_posts = sub_data.sort_values(by='score', ascending=False).head(10)
    else:
        top_posts = sub_data.sort_values(by='created_utc', ascending=False).head(10)
    
    posts_text = []
    for _, row in top_posts.iterrows():
        post = ""
        if 'title' in row and pd.notnull(row['title']):
            post += f"Title: {row['title']}\n"
        if 'selftext' in row and pd.notnull(row['selftext']):
            post += f"Content: {row['selftext']}\n"
        posts_text.append(post)
    
    combined_text = "\n---\n".join(posts_text)
    prompt = f"Please provide a concise summary for the following top 10 posts from the subreddit '{subreddit}':\n\n{combined_text}"
    return generate_summary_with_gemini(prompt)


st.title("Reddit Data Analysis Dashboard - Political Insights with Gemini LLM")

# Load data
file_path = "data.jsonl"
data = load_data(file_path)

# Summary Statistics
st.header("Summary Statistics")
total_posts = len(data)
if 'created_utc' in data.columns and not data['created_utc'].empty:
    min_date = data['created_utc'].min()
    max_date = data['created_utc'].max()
    date_range = f"{min_date.date()} to {max_date.date()}"
    st.write(f"**Total Posts:** {total_posts}")
    st.write(f"**Date Range:** {date_range}")
else:
    st.write("Timestamp data not available.")

if st.checkbox("Show available columns (Debug)"):
    st.write(data.columns.tolist())

# Overall Time Series Analysis
st.header("Overall Time Series Analysis")
query = st.text_input("Enter search query (searches title and selftext):")
time_series_data = extract_time_series(data, query)
if not time_series_data.empty:
    st.line_chart(time_series_data)
else:
    st.write("No data available for the specified query.")


st.header("Overall Keyword Analysis")
political_data = filter_political_posts(data, political_terms)




political_keywords = extract_keywords(political_data)
if political_keywords:
   
    
    st.subheader("Political Word Cloud")
    pol_wordcloud = generate_wordcloud(political_keywords)
    if pol_wordcloud:
        plt.figure(figsize=(10, 5))
        plt.imshow(pol_wordcloud.to_array(), interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
else:
    st.write("No political keyword data available.")


st.header("Posting Activity")
day_counts, hour_counts = post_activity(data)
if day_counts is not None and hour_counts is not None:
    st.subheader("Posts by Day of Week")
    plt.figure(figsize=(8, 4))
    sns.barplot(x=day_counts.index, y=day_counts.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Posts")
    plt.xlabel("Day of Week")
    st.pyplot(plt)
    
    st.subheader("Posts by Hour of Day")
    plt.figure(figsize=(8, 4))
    sns.barplot(x=hour_counts.index, y=hour_counts.values, palette="magma")
    plt.ylabel("Number of Posts")
    plt.xlabel("Hour of Day")
    st.pyplot(plt)
else:
    st.write("No posting activity data available.")


st.header("Top Authors and Subreddits")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Authors")
    authors = top_authors(data)
    if authors:
        authors_df = pd.DataFrame(authors, columns=["Author", "Count"]).set_index("Author")
        st.table(authors_df)
    else:
        st.write("No author data available.")
with col2:
    st.subheader("Top Subreddits")
    subreddits = top_subreddits(data)
    if subreddits:
        subreddits_df = pd.DataFrame(subreddits, columns=["Subreddit", "Count"]).set_index("Subreddit")
        st.table(subreddits_df)
    else:
        st.write("No subreddit data available.")


st.header("Top Shared Domains")
domains = top_shared_domains(data)
if domains:
    domains_df = pd.DataFrame(domains, columns=["Domain", "Count"]).set_index("Domain")
    st.table(domains_df)
else:
    st.write("No shared domain data available.")

# Overall Network Visualization
st.header("Network Visualization: Authors to Subreddits")
G = create_network(data)
if G.number_of_nodes() > 0:
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    community_map = detect_communities(G)
    communities = [community_map.get(node, 0) for node in G.nodes()]
    cmap = plt.cm.get_cmap('viridis', max(communities) + 1)
    nx.draw_networkx_nodes(G, pos, node_color=communities, cmap=cmap, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')
    st.pyplot(plt)
else:
    st.write("Network graph not available due to insufficient data.")


st.header("Political Analysis")
political_data = filter_political_posts(data, political_terms)
st.write(f"**Total Political Posts:** {len(political_data)}")




st.subheader("Political Sentiment Analysis")
if not political_data.empty:
    sentiment_df = political_sentiment_analysis(political_data)
    if not sentiment_df.empty:
        st.write("**Average Sentiment Scores (Political Posts):**")
        avg_sentiment = sentiment_df.mean()
        st.write(avg_sentiment)
        
        st.write("**Sentiment Distribution (Compound Score):**")
        plt.figure(figsize=(8, 4))
        sns.histplot(sentiment_df['compound'], bins=20, kde=True, color='skyblue')
        plt.xlabel("Compound Sentiment Score")
        plt.ylabel("Frequency")
        st.pyplot(plt)
    else:
        st.write("No sentiment data available.")
else:
    st.write("No political posts available for sentiment analysis.")

# st.subheader("Top Political Subreddits")
political_subreddits = top_subreddits(political_data)





st.header("Subreddit Topic Summarization")
subreddit_options = data['subreddit'].dropna().unique()
selected_subreddit = st.selectbox("Select a Subreddit for Summarization", subreddit_options)
if st.button("Summarize Top 10 Posts"):
    subreddit_summary = generate_subreddit_summary(selected_subreddit, data)
    st.markdown(subreddit_summary)

