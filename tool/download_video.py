import os
from yt_dlp import YoutubeDL

def download_youtube_video(url, save_path='./custom_data/videos'):
    try:
        os.makedirs(save_path, exist_ok=True)

        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4'
        }

        with YoutubeDL(ydl_opts) as ydl:
            print(f"🎬 Bắt đầu tải video từ: {url}")
            ydl.download([url])
            print(f"✅ Tải thành công! Video nằm trong thư mục: {save_path}")

    except Exception as e:
        print(f"❌ Lỗi khi tải video: {e}")

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=09R8_2nJtjg"
    download_youtube_video(video_url)
