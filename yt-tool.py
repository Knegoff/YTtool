import nltk

# Add the correct path where 'punkt' data is stored
nltk.data.path.append("/home/ec2-user/nltk_data")

# Ensure the 'punkt' tokenizer is available
nltk.download('punkt', download_dir="/home/ec2-user/nltk_data")



# Import necessary libraries
import streamlit as st  # Streamlit for building the UI
import googleapiclient.discovery  # To interact with YouTube API
import rake_nltk  # For keyword extraction
from collections import Counter  # To count keyword occurrences
import nltk  # Natural Language Toolkit (for downloading stopwords)
import pandas as pd  # To handle data in a DataFrame
import io  # To handle in-memory text streams for CSV
from googleapiclient.errors import HttpError  # Handle potential API errors

# Ensure stopwords are downloaded (important for RAKE)
nltk.download('stopwords')

# --- Helper Functions ---

# Function to search YouTube for the first 3 videos matching the keyword
def search_youtube_videos(keyword, api_key):
    # Initialize YouTube API client
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    # Request to search for videos based on the keyword
    request = youtube.search().list(
        q=keyword,  # The search query (keyword)
        part="snippet",  # What part of the video data to return
        maxResults=3,  # We want the top 3 results
        type="video"  # Search for videos only
    )
    response = request.execute()  # Execute the API request
    return response['items']  # Return the video data

# Function to extract video data (title, description, video ID)
def get_video_data(videos):
    video_data = []
    for video in videos:
        video_info = {
            "title": video['snippet']['title'],  # Video title
            "description": video['snippet']['description'],  # Video description
            "video_id": video['id']['videoId']  # Video ID
        }
        video_data.append(video_info)
    return video_data

# Function to extract key phrases from text using RAKE
def extract_keywords(text):
    rake = rake_nltk.Rake()  # Initialize RAKE for keyword extraction
    rake.extract_keywords_from_text(text)  # Extract keywords from the text
    return rake.get_ranked_phrases()  # Return the ranked phrases

# Function to summarize common keywords across multiple videos
def summarize_keywords(videos):
    # Initialize a counter to track keyword occurrences
    keyword_counter = Counter()

    # Loop through each video's description and extract keywords
    for video in videos:
        keywords = extract_keywords(video['description'])  # Extract keywords from each video description
        keyword_counter.update(keywords)  # Update the counter with new keywords
    
    # Return the most common keywords
    return keyword_counter.most_common()

# Function to generate CSV data from the video data and keyword summary
def generate_csv(video_data, common_keywords):
    # Create a DataFrame for video data
    video_df = pd.DataFrame(video_data)

    # Create a DataFrame for the summarized keywords
    keywords_df = pd.DataFrame(common_keywords, columns=['Keyword', 'Occurrences'])

    # Write to a CSV in memory (buffer)
    output = io.StringIO()  # Create an in-memory text stream
    video_df.to_csv(output, index=False, encoding='utf-8')  # Write video data to CSV
    output.write('\n')  # Add a line break between the tables
    keywords_df.to_csv(output, index=False, encoding='utf-8')  # Write keyword summary to CSV

    # Seek to the start of the buffer
    output.seek(0)
    
    return output  # Return the CSV output

# --- Streamlit UI Setup ---

# Set the title of the app
st.title("YouTube Video Keyword Analyzer")

# Description of the tool for users
st.write("""
This tool allows you to input a YouTube API key and a keyword to analyze the top 3 YouTube videos for that keyword.
It will extract the most common keywords from the video descriptions and allow you to download the results as a CSV.
""")

# Create text input fields for API key and keyword
api_key = st.text_input("Enter your YouTube API Key:", type="password")  # Mask the API key for security
keyword = st.text_input("Enter a Keyword to Search for YouTube Videos:")

# Create a button to trigger the analysis
if st.button("Analyze Videos"):
    # Check if both API key and keyword are provided
    if not api_key or not keyword:
        st.error("Please enter both the API key and a keyword!")
    else:
        try:
            # Perform the video search and analysis
            videos = search_youtube_videos(keyword, api_key)  # Search YouTube
            video_data = get_video_data(videos)  # Extract video data
            
            if not video_data:
                st.warning("No videos found for this keyword.")
            else:
                # Display video details to the user
                st.subheader(f"Top 3 Videos for Keyword: {keyword}")
                for video in video_data:
                    st.write(f"**Title:** {video['title']}")
                    st.write(f"**Description:** {video['description']}")
                    st.write(f"[Watch on YouTube](https://www.youtube.com/watch?v={video['video_id']})")
                    st.write("---")
                
                # Summarize the keywords from video descriptions
                common_keywords = summarize_keywords(video_data)
                
                # Display the keyword summary
                st.subheader("Common Keywords/Key Phrases:")
                if common_keywords:
                    for keyword, count in common_keywords:
                        st.write(f"{keyword}: {count} occurrences")
                else:
                    st.write("No keywords found.")
                
                # Generate the CSV file
                csv_output = generate_csv(video_data, common_keywords)
                
                # Provide a download button for the CSV
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output.getvalue(),
                    file_name=f'{keyword}_youtube_analysis.csv',
                    mime='text/csv'
                )

        except HttpError as e:
            # Handle any errors with the YouTube API
            st.error(f"An error occurred: {e}")

# Instructions on how to get a YouTube API key
st.write("""
**Don't have a YouTube API key?**
Follow the instructions [here](https://developers.google.com/youtube/registering_an_application) to get one.
""")
