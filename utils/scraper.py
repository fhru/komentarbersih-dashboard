import os
import re
import requests
from typing import List, Dict, Optional
import time
from html import unescape

def clean_html_text(text: str) -> str:
    """
    Membersihkan teks dari HTML tags.
    
    Args:
        text (str): Teks dengan HTML tags
        
    Returns:
        str: Teks yang sudah dibersihkan
    """
    if not text:
        return ""
    
    # Unescape HTML entities
    text = unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_video_id(url: str) -> Optional[str]:
    """
    Mengekstrak video ID dari URL YouTube.
    
    Args:
        url (str): URL video YouTube
        
    Returns:
        str: Video ID atau None jika tidak valid
    """
    # Pattern untuk berbagai format URL YouTube
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
        r'youtu\.be\/([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def get_video_info(video_id: str, api_key: str) -> Optional[Dict]:
    """
    Mendapatkan informasi video dari YouTube API.
    
    Args:
        video_id (str): ID video YouTube
        api_key (str): YouTube Data API key
        
    Returns:
        Dict: Informasi video atau None jika error
    """
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        'part': 'snippet,statistics',
        'id': video_id,
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data.get('items'):
            video_info = data['items'][0]
            return {
                'title': video_info['snippet']['title'],
                'channel': video_info['snippet']['channelTitle'],
                'view_count': video_info['statistics'].get('viewCount', 0),
                'comment_count': video_info['statistics'].get('commentCount', 0)
            }
        
    except requests.RequestException as e:
        print(f"Error saat mengambil info video: {e}")
    
    return None

def get_comments(video_id: str, api_key: str, max_results: int = 100) -> List[Dict]:
    """
    Mengambil komentar dari video YouTube.
    
    Args:
        video_id (str): ID video YouTube
        api_key (str): YouTube Data API key
        max_results (int): Jumlah maksimal komentar yang diambil
        
    Returns:
        List[Dict]: List komentar dengan informasi
    """
    comments = []
    next_page_token = None
    
    try:
        while len(comments) < max_results:
            url = "https://www.googleapis.com/youtube/v3/commentThreads"
            params = {
                'part': 'snippet',
                'videoId': video_id,
                'maxResults': min(100, max_results - len(comments)),
                'key': api_key,
                'order': 'relevance'
            }
            
            if next_page_token:
                params['pageToken'] = next_page_token
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                
                # Clean HTML from textDisplay
                clean_text = clean_html_text(comment['textDisplay'])
                
                comments.append({
                    'author': comment['authorDisplayName'],
                    'text': clean_text,
                    'like_count': comment['likeCount'],
                    'published_at': comment['publishedAt']
                })
            
            # Cek apakah ada halaman berikutnya
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
            
            # Delay untuk menghindari rate limiting
            time.sleep(0.1)
            
    except requests.RequestException as e:
        print(f"Error saat mengambil komentar: {e}")
    
    return comments[:max_results]

def get_api_key() -> Optional[str]:
    """
    Mendapatkan API key
    
    Returns:
        str: API key atau None jika tidak ditemukan
    """
    # Coba berbagai cara untuk mendapatkan API key
    api_key = None
    
    # dari environment variable
    api_key = os.getenv('YOUTUBE_API_KEY')
    if api_key:
        print("API key ditemukan dari environment variable")
        return api_key
    
    # Coba dari .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('YOUTUBE_API_KEY')
        if api_key:
            print("API key ditemukan dari .env file")
            return api_key
    except ImportError:
        print("dotenv tidak tersedia, skip loading .env file")
    
    print("❌ API key tidak ditemukan")
    print("Pastikan YOUTUBE_API_KEY tersedia di environment variable atau .env file")
    return None

def scrape_youtube_comments(video_url: str, max_comments: int = 100) -> Dict:
    """
    Scraping komentar dari video YouTube berdasarkan URL.
    
    Args:
        video_url (str): URL video YouTube
        max_comments (int): Jumlah maksimal komentar yang diambil
        
    Returns:
        Dict: Hasil scraping dengan info video dan komentar
    """
    # Ambil API key
    api_key = get_api_key()
    if not api_key:
        return {
            'success': False,
            'error': 'YouTube API key tidak ditemukan. Pastikan YOUTUBE_API_KEY tersedia di environment variable atau .env file.'
        }
    
    # Ekstrak video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        return {
            'success': False,
            'error': 'URL video YouTube tidak valid'
        }
    
    # Ambil info video
    video_info = get_video_info(video_id, api_key)
    if not video_info:
        return {
            'success': False,
            'error': 'Tidak dapat mengambil informasi video'
        }
    
    # Ambil komentar
    comments = get_comments(video_id, api_key, max_comments)
    
    return {
        'success': True,
        'video_id': video_id,
        'video_info': video_info,
        'comments': comments,
        'total_comments': len(comments)
    }

def get_comment_texts(video_url: str, max_comments: int = 100) -> List[str]:
    """
    Mengambil hanya teks komentar dari video YouTube.
    
    Args:
        video_url (str): URL video YouTube
        max_comments (int): Jumlah maksimal komentar yang diambil
        
    Returns:
        List[str]: List teks komentar
    """
    result = scrape_youtube_comments(video_url, max_comments)
    
    if not result['success']:
        print(f"Error: {result['error']}")
        return []
    
    # Ekstrak hanya teks komentar
    comment_texts = [comment['text'] for comment in result['comments']]
    return comment_texts

# Contoh penggunaan dan testing
if __name__ == "__main__":
    # Test HTML cleaning
    print("=== Test HTML Cleaning ===")
    test_html_texts = [
        "6:44 tertawa tapi terluka",
        "<a href=\"https://www.youtube.com/watch?v=bgJohjIX3ew&amp;t=404\">6:44</a> tertawa tapi terluka",
        "&lt;3 love this video!",
        "<b>Bold text</b> and <i>italic text</i>",
        "Normal text with <a href=\"#\">link</a>"
    ]
    
    for html_text in test_html_texts:
        cleaned = clean_html_text(html_text)
        print(f"Original: {html_text}")
        print(f"Cleaned:  {cleaned}")
        print()
    
    # Test API key
    print("=== Test API Key ===")
    api_key = get_api_key()
    if api_key:
        print(f"API Key: {api_key[:10]}...{api_key[-10:]}")
    else:
        print("API key tidak ditemukan")
    
    # Test URL parsing
    print("\n=== Test URL Parsing ===")
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s"
    ]
    
    for url in test_urls:
        video_id = extract_video_id(url)
        print(f"URL: {url}")
        print(f"Video ID: {video_id}")
        print()
    
    # Test scraping
    print("=== Test Scraping ===")
    test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        result = scrape_youtube_comments(test_video_url, max_comments=10)
        
        if result['success']:
            print(f"Video: {result['video_info']['title']}")
            print(f"Channel: {result['video_info']['channel']}")
            print(f"Total komentar: {result['total_comments']}")
            print("\nKomentar:")
            
            for i, comment in enumerate(result['comments'][:5], 1):
                print(f"{i}. {comment['author']}: {comment['text'][:100]}...")
        else:
            print(f"Error: {result['error']}")
            
    except Exception as e:
        print(f"Error saat testing: {e}")
