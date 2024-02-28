from flask import Flask, render_template, request, send_file
import os
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
from pytube import YouTube
import os
import threading
import time
import string
import random
import requests
from flask import Flask, request, jsonify, send_file, send_from_directory
from urllib.parse import unquote
import instaloader
# from pydub import AudioSegment
from flask import Flask, request, send_file, render_template_string
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
# from pydub import normalize
# from pydub import low_pass_filter
# from pydub import high_pass_filter
# from pydub import echos
# from pydub import split_on_silence
from google_auth_oauthlib.flow import InstalledAppFlow
from pydub import *
#import librosa
from flask import Flask, request, send_file
# from pydub import strip_silence
# from pydub import detect_silence
from flask import Flask, request, send_file
import os
from pytube import YouTube
import instaloader
from TikTokApi import *  # Corrected import statement
import requests
from bs4 import BeautifulSoup
from werkzeug.urls import unquote
from flask import Flask, request, send_file
from pytube import YouTube
from pydub import AudioSegment
from bs4 import BeautifulSoup
import requests
import instaloader
import os
from flask import Flask, request, send_file, abort
from pytube import YouTube
from pydub import AudioSegment
from bs4 import BeautifulSoup
import requests
import instaloader
import os
from flask import Flask, request, send_file, jsonify
import os
import threading
import time
import requests
import random
import string
import time
import threading
import pytube
from pytube import *
from flask import Flask, request, send_file
from pytube import YouTube
from pydub import AudioSegment
from bs4 import BeautifulSoup
import requests
import instaloader
import os
import threading
import time
from flask import Flask, request, send_file, jsonify
from pytube import YouTube
from pydub import AudioSegment
from bs4 import BeautifulSoup
import requests
import instaloader
import os
import threading
import time
from flask import Flask, request, send_file, jsonify
from pydub import AudioSegment
from bs4 import BeautifulSoup
import requests
import os
import threading
import time
from flask import Flask, request, send_file, jsonify
from pydub import AudioSegment
from bs4 import BeautifulSoup
import requests
import os
import threading
import time
from urllib.parse import unquote
from urllib.parse import *
from urllib.parse import quote
from flask import Flask, request, send_file, jsonify
import os
import threading
import time
import requests
from urllib.parse import unquote
import os
import threading
import time
import string
import random
from flask import Flask, request, jsonify, send_file
from urllib.parse import unquote
import yt_dlp
import instaloader
from TikTokApi import TikTokapi
import ffmpeg
import os
import threading
import time
import string
import random
from flask import Flask, request, jsonify, send_file
from urllib.parse import unquote
import yt_dlp
import instaloader
from pydub import AudioSegment
import ffmpeg




app = Flask(__name__)

# Configuración de las credenciales de Spotify
os.environ["SPOTIPY_CLIENT_ID"] = "TU_CLIENT_ID_DE_SPOTIFY"
os.environ["SPOTIPY_CLIENT_SECRET"] = "TU_CLIENT_SECRET_DE_SPOTIFY"
os.environ["SPOTIPY_REDIRECT_URI"] = "http://localhost:5000/callback"

# Configuración de las credenciales de YouTube
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'




@app.route("/")
def index():
    return '''
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }

        header {
            background-color: #333;
            padding: 15px;
            text-align: center;
        }

        header a {
            color: #fff;
            text-decoration: none;
            margin: 0 15px;
            font-size: 18px;
            font-weight: bold;
        }

        .container {
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
            max-width: 400px;
            width: 100%;
            margin: 20px auto;
        }

        h1 {
            color: #333;
            margin-bottom: 30px;
        }

        form {
            margin-top: 20px;
        }

        input[type="text"],
        input[type="file"],
        select {
            padding: 10px;
            margin: 10px 0;
            width: calc(100% - 20px);
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
        }

        input[type="submit"] {
            padding: 12px 24px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>

<body>
    <header>
        <a href="#transfer">Transferir</a>
        <a href="#download">Descargar</a>
        <a href="#process_audio">Procesar Audio</a>
    </header>

    <div class="container" id="transfer">
        <h1>Transferir Spotify A Youtube</h1>
        <form action="/transfer" method="get">
            <label for="spotify_playlist_id">Spotify Playlist ID:</label><br>
            <input type="text" id="spotify_playlist_id" name="spotify_playlist_id"
                placeholder="Ingrese el ID de la lista de reproducción de Spotify" value=""><br>
            <input type="submit" value="Transferir a YouTube">
        </form>
    </div>

    <div class="container" id="download">
        <h1>Media Downloader</h1>
        <form action="/download" method="get" id="downloadForm">
            <label for="content_url">URL del contenido:</label><br>
            <input type="text" id="content_url" name="content_url" placeholder="Ingrese la URL del contenido" value=""><br>
            <label for="download_type">Tipo de descarga:</label><br>
            <select id="download_type" name="download_type">
                <option value="video">Video</option>
                <option value="audio">Audio</option>
            </select><br>
            <label for="quality">Calidad (solo para video):</label><br>
            <select id="quality" name="quality">
                <option value="max">Máxima disponible</option>
                <option value="4k">4K</option>
                <option value="720p">720p</option>
                <option value="480p">480p</option>
            </select><br>
            <input type="submit" value="Descargar">
        </form>
    </div>

    <div class="container" id="process_audio">
        <h1>Edición De Audio o Video</h1>
        <form action="/process_audio" method="post" enctype="multipart/form-data">
            <input type="file" name="audio_file">
            <input type="submit" value="Procesar Video o Audio">
        </form>
    </div>

    <script>
        // Decodificar todas las URL y campos al cargar la página
        window.onload = function () {
            var params = new URLSearchParams(window.location.search);

            function decodeAndSetValue(elementId, paramName) {
                var value = params.get(paramName);
                if (value !== null) {
                    document.getElementById(elementId).value = decodeURIComponent(value);
                }
            }

            // Decodificar Spotify Playlist ID
            decodeAndSetValue('spotify_playlist_id', 'spotify_playlist_id');

            // Decodificar URL del contenido
            decodeAndSetValue('content_url', 'content_url');

            // Decodificar Tipo de descarga
            decodeAndSetValue('download_type', 'download_type');

            // Decodificar Calidad
            decodeAndSetValue('quality', 'quality');
        };

        function updateDownloadFormAction() {
            var form = document.getElementById('downloadForm');
            var contentUrl = encodeURIComponent(document.getElementById('content_url').value);
            var downloadType = encodeURIComponent(document.getElementById('download_type').value);
            var quality = encodeURIComponent(document.getElementById('quality').value);
            form.action = '/download?' + 'content_url=' + contentUrl + '&download_type=' + downloadType + '&quality=' + quality;
        }

        document.getElementById('content_url').addEventListener('input', updateDownloadFormAction);

        document.getElementById('downloadForm').addEventListener('submit', function (event) {
            var form = event.target;
            var contentUrl = encodeURIComponent(document.getElementById('content_url').value);
            var downloadType = encodeURIComponent(document.getElementById('download_type').value);
            var quality = encodeURIComponent(document.getElementById('quality').value);
            form.action = '/download?' + 'content_url=' + contentUrl + '&download_type=' + downloadType + '&quality=' + quality;
        });
    </script>
</body>

</html>

'''
    if request.method == "GET":
        if "file" not in request.files:
            return "No se seleccionó ningún archivo."
        file = request.files["file"]
        if file.filename == "":
            return "No se seleccionó ningún archivo."
        filename = file.filename
        file.save(filename)
        cleaned_filename = clean_audio_video(filename)
        os.remove(filename)
        return send_file(cleaned_filename, as_attachment=True)


@app.route("/transfer", methods=["GET"])
def transfer():
    spotify_playlist_id = request.form["spotify_playlist_id"]
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="playlist-read-private"))
    tracks = get_spotify_playlist_tracks(sp, spotify_playlist_id)
    youtube = get_youtube_authenticated_service()
    youtube_playlist_id = create_youtube_playlist(
        youtube, "Nueva Playlist", "Descripción de la nueva playlist"
    )
    for track in tracks:
        track_name = track["track"]["name"]
        artist_name = track["track"]["artists"][0]["name"]
        query = f"{track_name} {artist_name} audio"
        request = youtube.search().list(part="snippet", maxResults=1, q=query)
        response = request.execute()
        video_id = response["items"][0]["id"]["videoId"]
        add_video_to_playlist(youtube, video_id, youtube_playlist_id)




TEMP_DIR = "temp_files"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

file_cleanup_lock = threading.Lock()

def file_cleanup():
    while True:
        time.sleep(60)  # Check every 60 seconds
        with file_cleanup_lock:
            for file_name in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file_name)
                if os.path.isfile(file_path) and time.time() - os.path.getctime(file_path) > 300:
                    os.remove(file_path)

file_cleanup_thread = threading.Thread(target=file_cleanup)
file_cleanup_thread.daemon = True
file_cleanup_thread.start()

def generate_random_name(length=10):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def move_file(src, dest):
    try:
        os.rename(src, dest)
        return True
    except Exception as e:
        print(f"Error moving file: {e}")
        return False

def delete_file(file_path):
    try:
        os.remove(file_path)
        return True
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False

def edit_file(file_path, new_content):
    try:
        with open(file_path, 'w') as file:
            file.write(new_content)
        return True
    except Exception as e:
        print(f"Error editing file: {e}")
        return False

def import_file(src, dest):
    try:
        shutil.copyfile(src, dest)
        return True
    except Exception as e:
        print(f"Error importing file: {e}")
        return False

def create_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error creating file: {e}")
        return False

def download_youtube_video(url):
    try:
        ydl_opts = {
            'outtmpl': os.path.join(TEMP_DIR, f'{generate_random_name()}_video_youtube.%(ext)s'),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info['title'], ydl.prepare_filename(info)
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return None, None

def download_youtube_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(TEMP_DIR, f'{generate_random_name()}_audio_youtube.%(ext)s'),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info['title'], ydl.prepare_filename(info)
    except Exception as e:
        print(f"Error downloading YouTube audio: {e}")
        return None, None

def download_instagram_video(url):
    try:
        loader = instaloader.Instaloader()
        post = instaloader.Post.from_shortcode(loader.context, url)
        video_url = post.url
        file_path = os.path.join(TEMP_DIR, f'{generate_random_name()}_video_instagram.mp4')
        os.system(f'wget {video_url} -O {file_path}')
        return f'{generate_random_name()}_video_instagram.mp4', file_path
    except Exception as e:
        print(f"Error downloading Instagram video: {e}")
        return None, None

def download_tiktok_video(url):
    try:
        api = TikTokApi()
        video = api.get_video_by_url(url)
        video_url = video['itemInfo']['itemStruct']['video']['downloadAddr']
        file_path = os.path.join(TEMP_DIR, f'{generate_random_name()}_video_tiktok.mp4')
        os.system(f'wget {video_url} -O {file_path}')
        return f'{generate_random_name()}_video_tiktok.mp4', file_path
    except Exception as e:
        print(f"Error downloading TikTok video: {e}")
        return None, None

def get_file_extension(content_type):
    if 'audio' in content_type:
        return '.mp3'
    elif 'video' in content_type:
        return '.mp4'
    else:
        return '.txt'

def download_file(url, download_type):
    try:
        response = requests.get(url)
        file_path = os.path.join(TEMP_DIR, f'{generate_random_name()}_file{get_file_extension(download_type)}')
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return f'{generate_random_name()}_file{get_file_extension(download_type)}', file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None, None

def convert_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        converted_path = file_path.replace('.mp3', '_converted.mp3')
        audio.export(converted_path, format='mp3')
        return f'{generate_random_name()}_converted.mp3', converted_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None, None


@app.route('/download', methods=['GET'])
def download():
    content_url = unquote(request.args.get("content_url"))
    download_type = request.args.get("download_type")
    
    if download_type == 'video':
        if 'youtube.com' in content_url:
            video_title, file_path = download_youtube_video(content_url)
        elif 'instagram.com' in content_url:
            video_title, file_path = download_instagram_video(content_url)
        elif 'tiktok.com' in content_url:
            video_title, file_path = download_tiktok_video(content_url)
        else:
            return jsonify({"error": "Invalid content URL for video download."}), 400
    elif download_type == 'audio':
        if 'youtube.com' in content_url:
            audio_title, file_path = download_youtube_audio(content_url)
            if file_path:
                new_audio_title, file_path = convert_audio(file_path)
        else:
            return jsonify({"error": "Invalid content URL for audio download."}), 400
    else:
        return jsonify({"error": "Invalid download type."}), 400

    if file_path:
        file_name = os.path.basename(file_path)
        new_file_path = os.path.join(TEMP_DIR, f'{generate_random_name()}_{video_title or audio_title}{get_file_extension(download_type)}')
        
        if move_file(file_path, new_file_path):
            return send_file(new_file_path, as_attachment=True, download_name=f'{video_title or audio_title}{get_file_extension(download_type)}')
        else:
            return jsonify({"error": "Error moving file."}), 500

    return jsonify({"error": "Error downloading the file."}), 500






def get_spotify_playlist_tracks(sp, playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results["items"]
    while results["next"]:
        results = sp.next(results)
        tracks.extend(results["items"])
    return tracks


def get_youtube_authenticated_service():
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, SCOPES
    )
    credentials = flow.run_console()
    return googleapiclient.discovery.build(
        API_SERVICE_NAME, API_VERSION, credentials=credentials
    )


def create_youtube_playlist(youtube, title, description):
    request = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {"title": title, "description": description},
            "status": {"privacyStatus": "private"},
        },
    )
    response = request.execute()
    return response["id"]
    request = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {"title": title, "description": description},
            "status": {"privacyStatus": "private"},
        },
    )
    response = request.execute()
    return response["id"]


def add_video_to_playlist(youtube, video_id, playlist_id):
    request = youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "position": 0,
                "resourceId": {"kind": "youtube", "videoId": video_id},
            }
        },
    )
    response = request.execute()
    return response



@app.route("/edit_audio", methods=["GET"])
def edit_audio():
    file = request.files["audio_file"]
    file_path = "temp_audio.wav"
    file.save(file_path)
    audio = AudioSegment.from_wav(file_path)
    audio_filtered = filter_silence(audio, silence_threshold=(-40))
    edited_file_path = "edited_audio.wav"
    audio_filtered.export(edited_file_path, format="wav")
    os.remove(file_path)
    return send_file(edited_file_path, as_attachment=True)
    audio_file = request.files["file"]
    edited_file_path = os.path.join("edited_files", audio_file.filename)
    audio = AudioSegment.from_file(audio_file)
    audio = audio.low_pass_filter(1000)
    audio.export(edited_file_path, format="mp3")
    uploaded_file = request.files["file"]
    file_extension = uploaded_file.filename.split(".")[(-1)]
    if file_extension in ["mp3", "wav"]:
        clean_file = clean_audio(uploaded_file)
        return send_file(clean_file, as_attachment=True)
    else:
        return "El archivo debe ser un archivo de audio en formato MP3 o WAV."
    sound = AudioSegment.from_file(file_path)
    cleaned_sound = sound.low_pass_filter(1000)
    cleaned_sound.export(clean_file_path, format="mp3")
    sound.close()
    cleaned_sound.close()
    return clean_file_path
    if "audio_file" not in request.files:
        return "No se encontró ningún archivo en la solicitud."
    audio_file = request.files["audio_file"]
    if audio_file.filename == "":
        return "No se seleccionó ningún archivo."
    if audio_file:
        filename = secure_filename(audio_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio_file.save(file_path)
        sound = AudioSegment.from_file(file_path)
        edited_sound = sound.low_pass_filter(1000)
        edited_sound = edited_sound.high_pass_filter(500)
        edited_file_path = os.path.join(
            app.config["CLEAN_FOLDER"], ("edited_" + filename)
        )
        edited_sound.export(edited_file_path, format="wav")
        return send_file(edited_file_path, as_attachment=True)
    return "Error al procesar el archivo."


def filter_silence(audio, silence_threshold=(-40)):
    segments = split_on_silence(audio, silence_thresh=silence_threshold)
    audio_filtered = AudioSegment.silent()
    for segment in segments:
        audio_filtered += segment
    return audio_filtered
    segments = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        keep_silence=keep_silence,
    )
    filtered_audio = AudioSegment.empty()
    for segment in segments:
        filtered_audio += segment
    return filtered_audio
    chunks = split_on_silence(
        audio, min_silence_len=1000, silence_thresh=silence_threshold)
    silent_chunks = [chunk for chunk in chunks if (chunk.dBFS < silence_threshold)]
    for silent_chunk in silent_chunks:
        audio = audio.remove(silent_chunk)
    return audio



def clean_audio(file_path):
    (audio_data, sr) = librosa.load(file_path)
    audio_data = librosa.effects.trim(audio_data, top_db=30)[0]
    clean_file_path = "clean_" + os.path.basename(file_path)
    librosa.output.write_wav(clean_file_path, audio_data, sr)
    return clean_file_path
    audio = AudioSegment.from_file(uploaded_file)
    clean_audio = audio.low_pass_filter(3000).high_pass_filter(300)
    clean_file = "clean_audio." + uploaded_file.filename.split(".")[(-1)]
    clean_audio.export(clean_file, format=uploaded_file.filename.split(".")[(-1)])
    return clean_file
    uploaded_file = request.files["audio_file"]
    if uploaded_file.filename != "":
        file_path = os.path.join("uploads", uploaded_file.filename)
        uploaded_file.save(file_path)
        audio = AudioSegment.from_file(file_path)
        clean_audio = (
            audio.low_pass_filter(2000).high_pass_filter(200).fade_in(100).fade_out(100)
        )
        clean_file_path = os.path.join("cleaned", uploaded_file.filename)
        clean_audio.export(clean_file_path, format="mp3")
        return send_file(clean_file_path, as_attachment=True)
    return "No se ha proporcionado ningún archivo de audio."
    if "file" not in request.files:
        return "No se encontró ningún archivo en la solicitud."
    file = request.files["file"]
    if file.filename == "":
        return "No se seleccionó ningún archivo."
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        cleaned_filename = clean_audio_video(file_path)
        return send_file(cleaned_filename, as_attachment=True)
    return "Error al procesar el archivo."
    (y, sr) = librosa.load(file_path)
    y_filt = librosa.effects.trim(y)
    y_filt = librosa.util.normalize(y_filt[0])
    return (y_filt, sr)
    (audio, sr) = librosa.load(file_path)
    audio = librosa.effects.remix(audio, intervals=librosa.effects.split(audio))
    return (audio, sr)


def clean_audio_video(filename):
    (_, file_extension) = os.path.splitext(filename)
    if file_extension.lower() in [".mp3", ".wav"]:
        audio = AudioSegment.from_file(filename)
        cleaned_audio = normalize(audio)
        cleaned_filename = f"cleaned_{filename}"
        cleaned_audio.export(cleaned_filename, format=file_extension[1:])
    else:
        return "Formato de archivo no compatible. Por favor, seleccione un archivo de audio en formato MP3 o WAV."
    return cleaned_filename
    (filename, extension) = os.path.splitext(file_path)
    if (extension == ".mp3") or (extension == ".wav"):
        audio = AudioSegment.from_file(file_path)
        audio = normalize(audio)
        audio = audio.set_frame_rate(44100)
        audio = audio.set_sample_width(2)
        audio.export((filename + "_cleaned.wav"), format="wav")
        return filename + "_cleaned.wav"
    elif extension == ".mp4":
        video = VideoFileClip(file_path)
        audio = video.audio
        audio.write_audiofile((filename + "_audio.wav"))
        cleaned_audio = clean_audio((filename + "_audio.wav"))
        video.set_audio(cleaned_audio)
        video.write_videofile((filename + "_cleaned.mp4"), codec="libx264")
        return filename + "_cleaned.mp4"
        return "Formato de archivo no compatible."
    video = VideoFileClip(file_path)
    audio = video.audio
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = normalize(audio)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=(-36))
    silent_chunks = [chunk for chunk in chunks if (chunk.dBFS < (-36))]
    for silent_chunk in silent_chunks:
        audio = audio.remove(silent_chunk)
    cleaned_file_path = file_path.replace("uploads", "cleaned")
    audio.export(cleaned_file_path, format="wav")
    return cleaned_file_path


@app.route("/clean", methods=["GET"])
def clean():
    if "file" not in request.files:
        return "No se seleccionó ningún archivo."
    file = request.files["file"]
    if file.filename == "":
        return "No se encontró ningún archivo en la solicitud."
    filename = file.filename
    file.save(filename)
    cleaned_filename = clean_audio_video(filename)
    os.remove(filename)
    return send_file(cleaned_filename, as_attachment=True)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        cleaned_filename = clean_audio_video(file_path)
        return send_file(cleaned_filename, as_attachment=True)
    return "Error al procesar el archivo."


def edit_audio_file(file_path):
    sound = AudioSegment.from_file(file_path)
    edited_sound = sound.low_pass_filter(5000)
    edited_file_path = "edited_" + file_path
    edited_sound.export(edited_file_path, format="mp3")
    return edited_file_path


@app.route("/edit", methods=["GET"])
def edit_file():
    if "file" not in request.files:
        return "No se ha proporcionado ningún archivo"
    file = request.files["file"]
    if file.filename == "":
        return "No se ha seleccionado ningún archivo"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    clean_file_path = os.path.join(CLEAN_FOLDER, f"cleaned_{file.filename}")
    file.save(file_path)
    if file_path.endswith(".mp4") or file_path.endswith(".mov"):
        clean_file_path = edit_video(file_path, clean_file_path)
    elif file_path.endswith(".mp3") or file_path.endswith(".wav"):
        clean_file_path = edit_audio(file_path, clean_file_path)
    else:
        return "Tipo de archivo no compatible"
    return send_file(clean_file_path, as_attachment=True)


def edit_video(file_path, clean_file_path):
    clip = VideoFileClip(file_path)
    edited_clip = clip.subclip()
    edited_clip.write_videofile(clean_file_path, codec="libx264", audio_codec="aac")
    clip.close()
    edited_clip.close()
    os.remove(file_path)
    return clean_file_path


@app.route("/remove_noise", methods=["GET"])
def remove_noise():
    if "noisy_file" not in request.files:
        return "No se encontró ningún archivo en la solicitud."
    noisy_file = request.files["noisy_file"]
    if noisy_file.filename == "":
        return "No se seleccionó ningún archivo."
    if noisy_file:
        filename = secure_filename(noisy_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        noisy_file.save(file_path)
        (cleaned_audio, sr) = clean_audio(file_path)
        cleaned_file_path = os.path.join(
            app.config["CLEAN_FOLDER"], ("cleaned_" + filename)
        )
        librosa.output.write_wav(cleaned_file_path, cleaned_audio, sr)
        os.remove(file_path)
        return send_file(cleaned_file_path, as_attachment=True)
    return "Error al procesar el archivo."


@app.route("/normalize_volume", methods=["GET"])
def normalize_volume():
    if "volume_file" not in request.files:
        return "No se encontró ningún archivo en la solicitud."
    volume_file = request.files["volume_file"]
    if volume_file.filename == "":
        return "No se seleccionó ningún archivo."
    if volume_file:
        filename = secure_filename(volume_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        volume_file.save(file_path)
        normalized_audio = normalize_audio(file_path)
        normalized_file_path = os.path.join(
            app.config["CLEAN_FOLDER"], ("normalized_" + filename)
        )
        normalized_audio.export(normalized_file_path, format="wav")
        return send_file(normalized_file_path, as_attachment=True)
    return "Error al procesar el archivo."


def normalize_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    normalized_audio = normalize(audio)
    return normalized_audio

@app.route("/remove_echo", methods=["GET"])
def remove_echo():
    if "echo_file" not in request.files:
        return "No se encontró ningún archivo en la solicitud."
    echo_file = request.files["echo_file"]
    if echo_file.filename == "":
        return "No se seleccionó ningún archivo."
    if echo_file:
        filename = secure_filename(echo_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        echo_file.save(file_path)
        sound = AudioSegment.from_file(file_path)
        cleaned_sound = echos(sound, delay=50, decays=0.2)
        cleaned_file_path = os.path.join(
            app.config["CLEAN_FOLDER"], ("cleaned_" + filename)
        )
        cleaned_sound.export(cleaned_file_path, format="wav")
        return send_file(cleaned_file_path, as_attachment=True)
    return "Error al procesar el archivo."


@app.route("/process_audio", methods=["GET"])
def process_audio():
    if "audio_file" not in request.files:
        return "No se encontró ningún archivo de audio en la solicitud."
    audio_file = request.files["audio_file"]
    if audio_file.filename == "":
        return "No se seleccionó ningún archivo de audio."
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    audio_file.save(file_path)
    sound = AudioSegment.from_file(file_path)
    cleaned_sound = echos(sound, delay=50, decays=0.2)
    cleaned_sound = normalize(cleaned_sound)
    (audio, sr) = librosa.load(file_path)
    audio = librosa.effects.remix(audio, intervals=librosa.effects.split(audio))
    cleaned_audio = librosa.output.write_wav(cleaned_sound, audio, sr)
    video = VideoFileClip(file_path)
    audio = video.audio
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = normalize(audio)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=(-36))
    silent_chunks = [chunk for chunk in chunks if (chunk.dBFS < (-36))]
    for silent_chunk in silent_chunks:
        audio = audio.remove(silent_chunk)
    cleaned_file_path = os.path.join(
        app.config["CLEAN_FOLDER"], ("cleaned_" + filename)
    )
    audio.export(cleaned_file_path, format="wav")
    return send_file(cleaned_file_path, as_attachment=True)
    cleaned_file_path = process_audio_file(audio_file)
    audio = strip_silence(audio, silence_len=1000, silence_thresh=(-30))


@app.route("/process_audio_form")
def process_audio_form():
    return '''
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
        }

        header {
            background-color: #333;
            padding: 15px;
            text-align: center;
        }

        header a {
            color: #fff;
            text-decoration: none;
            margin: 0 15px;
            font-size: 18px;
            font-weight: bold;
        }

        .container {
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
            max-width: 400px;
            width: 100%;
            margin: 20px auto;
        }

        h1 {
            color: #333;
            margin-bottom: 30px;
        }

        form {
            margin-top: 20px;
        }

        input[type="text"],
        input[type="file"],
        select {
            padding: 10px;
            margin: 10px 0;
            width: calc(100% - 20px);
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
        }

        input[type="submit"] {
            padding: 12px 24px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>

<body>
    <header>
        <a href="#transfer">Transferir</a>
        <a href="#download">Descargar</a>
        <a href="#process_audio">Procesar Audio</a>
    </header>

    <div class="container" id="transfer">
        <h1>Transferir Spotify A Youtube</h1>
        <form action="/transfer" method="get">
            <label for="spotify_playlist_id">Spotify Playlist ID:</label><br>
            <input type="text" id="spotify_playlist_id" name="spotify_playlist_id"
                placeholder="Ingrese el ID de la lista de reproducción de Spotify" value=""><br>
            <input type="submit" value="Transferir a YouTube">
        </form>
    </div>

    <div class="container" id="download">
        <h1>Media Downloader</h1>
        <form action="/download" method="get" id="downloadForm">
            <label for="content_url">URL del contenido:</label><br>
            <input type="text" id="content_url" name="content_url" placeholder="Ingrese la URL del contenido" value=""><br>
            <label for="download_type">Tipo de descarga:</label><br>
            <select id="download_type" name="download_type">
                <option value="video">Video</option>
                <option value="audio">Audio</option>
            </select><br>
            <label for="quality">Calidad (solo para video):</label><br>
            <select id="quality" name="quality">
                <option value="max">Máxima disponible</option>
                <option value="4k">4K</option>
                <option value="720p">720p</option>
                <option value="480p">480p</option>
            </select><br>
            <input type="submit" value="Descargar">
        </form>
    </div>

    <div class="container" id="process_audio">
        <h1>Edición De Audio o Video</h1>
        <form action="/process_audio" method="post" enctype="multipart/form-data">
            <input type="file" name="audio_file">
            <input type="submit" value="Procesar Video o Audio">
        </form>
    </div>

    <script>
        // Decodificar todas las URL y campos al cargar la página
        window.onload = function () {
            var params = new URLSearchParams(window.location.search);

            function decodeAndSetValue(elementId, paramName) {
                var value = params.get(paramName);
                if (value !== null) {
                    document.getElementById(elementId).value = decodeURIComponent(value);
                }
            }

            // Decodificar Spotify Playlist ID
            decodeAndSetValue('spotify_playlist_id', 'spotify_playlist_id');

            // Decodificar URL del contenido
            decodeAndSetValue('content_url', 'content_url');

            // Decodificar Tipo de descarga
            decodeAndSetValue('download_type', 'download_type');

            // Decodificar Calidad
            decodeAndSetValue('quality', 'quality');
        };

        function updateDownloadFormAction() {
            var form = document.getElementById('downloadForm');
            var contentUrl = encodeURIComponent(document.getElementById('content_url').value);
            var downloadType = encodeURIComponent(document.getElementById('download_type').value);
            var quality = encodeURIComponent(document.getElementById('quality').value);
            form.action = '/download?' + 'content_url=' + contentUrl + '&download_type=' + downloadType + '&quality=' + quality;
        }

        document.getElementById('content_url').addEventListener('input', updateDownloadFormAction);

        document.getElementById('downloadForm').addEventListener('submit', function (event) {
            var form = event.target;
            var contentUrl = encodeURIComponent(document.getElementById('content_url').value);
            var downloadType = encodeURIComponent(document.getElementById('download_type').value);
            var quality = encodeURIComponent(document.getElementById('quality').value);
            form.action = '/download?' + 'content_url=' + contentUrl + '&download_type=' + downloadType + '&quality=' + quality;
        });
    </script>
</body>

</html>

'''

def process_audio_file(
    file_path, audio_file, clean_file_path, filename, audio, silence_threshold=(-40)
):
    filename = secure_filename(audio_file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    audio_file.save(file_path)
    sound = AudioSegment.from_file(file_path)
    cleaned_sound = echos(sound, delay=50, decays=0.2)
    if sound.dBFS > (-10):
        cleaned_sound = normalize(cleaned_sound)
    (audio, sr) = librosa.load(file_path)
    audio = librosa.effects.remix(audio, intervals=librosa.effects.split(audio))
    cleaned_audio = librosa.output.write_wav(cleaned_sound, audio, sr)
    video = VideoFileClip(file_path)
    audio = video.audio
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = normalize(audio)
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=(-36))
    silent_chunks = [chunk for chunk in chunks if (chunk.dBFS < (-36))]
    for silent_chunk in silent_chunks:
        audio = audio.remove(silent_chunk)
    audio = strip_silence(audio, silence_len=1000, silence_thresh=(-30))
    cleaned_file_path = os.path.join(
        app.config["CLEAN_FOLDER"], ("cleaned_" + filename)
    )
    audio.export(cleaned_file_path, format="wav")
    return cleaned_file_path
    audio = audio.low_pass_filter(5000)
    if "audio_file" not in request.files:
        return "No se encontró ningún archivo de audio en la solicitud."
    audio_file = request.files["audio_file"]
    if audio_file.filename == "":
        return "No se seleccionó ningún archivo de audio."
    cleaned_sound = normalize(cleaned_sound)
    return send_file(cleaned_file_path, as_attachment=True)
    cleaned_file_path = process_audio_file(audio_file)
    if "echo_file" not in request.files:
        return "No se encontró ningún archivo en la solicitud."
    echo_file = request.files["echo_file"]
    if echo_file.filename == "":
        return "No se seleccionó ningún archivo."
    if echo_file:
        filename = secure_filename(echo_file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        echo_file.save(file_path)
        sound = AudioSegment.from_file(file_path)
        cleaned_sound = echos(sound, delay=50, decays=0.2)
        cleaned_file_path = os.path.join(
            app.config["CLEAN_FOLDER"], ("cleaned_" + filename)
        )
        cleaned_sound.export(cleaned_file_path, format="wav")
        return send_file(cleaned_file_path, as_attachment=True)
    return "Error al procesar el archivo."
    if "volume_file" not in request.files:
        return "No se encontró ningún archivo en la solicitud."
    volume_file = request.files["volume_file"]
    if volume_file.filename == "":
        return "No se seleccionó ningún archivo."
    if volume_file:
        filename = secure_filename(volume_file.filename)
        volume_file.save(file_path)
        normalized_audio = normalize_audio(file_path)
        normalized_file_path = os.path.join(
            app.config["CLEAN_FOLDER"], ("normalized_" + filename)
            )
        normalized_audio.export(normalized_file_path, format="wav")
        return send_file(normalized_file_path, as_attachment=True)
    clip = VideoFileClip(file_path)
    edited_clip = clip.subclip()
    edited_clip.write_videofile(clean_file_path, codec="libx264", audio_codec="aac")
    clip.close()
    edited_clip.close()
    os.remove(file_path)
    return clean_file_path
    if "noisy_file" not in request.files:
        return "No se encontró ningún archivo en la solicitud."
    noisy_file = request.files["noisy_file"]
    if noisy_file.filename == "":
        return "No se seleccionó ningún archivo."
    if noisy_file:
        filename = secure_filename(noisy_file.filename)
        noisy_file.save(file_path)
        (cleaned_audio, sr) = clean_audio(file_path)
        librosa.output.write_wav(cleaned_file_path, cleaned_audio, sr)
        os.remove(file_path)
    if "file" not in request.files:
        return "No se ha proporcionado ningún archivo"
    file = request.files["file"]
    if file.filename == "":
        return "No se ha seleccionado ningún archivo"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    clean_file_path = os.path.join(CLEAN_FOLDER, f"cleaned_{file.filename}")
    file.save(file_path)
    if file_path.endswith(".mp4") or file_path.endswith(".mov"):
        clean_file_path = edit_video(file_path, clean_file_path)
    elif file_path.endswith(".mp3") or file_path.endswith(".wav"):
        clean_file_path = edit_audio(file_path, clean_file_path)
    else:
        return "Tipo de archivo no compatible"
    return send_file(clean_file_path, as_attachment=True)
    audio = AudioSegment.from_file(file_path)
    normalized_audio = normalize(audio)
    return normalized_audio
    edited_sound = sound.low_pass_filter(5000)
    edited_file_path = "edited_" + file_path
    edited_sound.export(edited_file_path, format="mp3")
    return edited_file_path
    filename = file.filename
    file.save(filename)
    cleaned_filename = clean_audio_video(filename)
    os.remove(filename)
    return send_file(cleaned_filename, as_attachment=True)
    if file:
        filename = secure_filename(file.filename)
        file.save(file_path)
        cleaned_filename = clean_audio_video(file_path)
        return send_file(cleaned_filename, as_attachment=True)
    (_, file_extension) = os.path.splitext(filename)
    if file_extension.lower() in [".mp3", ".wav"]:
        audio = AudioSegment.from_file(filename)
        cleaned_audio = normalize(audio)
        cleaned_filename = f"cleaned_{filename}"
        cleaned_audio.export(cleaned_filename, format=file_extension[1:])
        return "Formato de archivo no compatible. Por favor, seleccione un archivo de audio en formato MP3 o WAV."
    return cleaned_filename
    (filename, extension) = os.path.splitext(file_path)
    if (extension == ".mp3") or (extension == ".wav"):
        audio = AudioSegment.from_file(file_path)
        audio = normalize(audio)
        audio = audio.set_frame_rate(44100)
        audio = audio.set_sample_width(2)
        audio.export((filename + "_cleaned.wav"), format="wav")
        return filename + "_cleaned.wav"
    elif extension == ".mp4":
        video = VideoFileClip(file_path)
        audio = video.audio
        audio.write_audiofile((filename + "_audio.wav"))
        cleaned_audio = clean_audio((filename + "_audio.wav"))
        video.set_audio(cleaned_audio)
        video.write_videofile((filename + "_cleaned.mp4"), codec="libx264")
        return filename + "_cleaned.mp4"
        return "Formato de archivo no compatible."
    cleaned_file_path = file_path.replace("uploads", "cleaned")
    segments = split_on_silence(audio, silence_thresh=silence_threshold)
    audio_filtered = AudioSegment.silent()
    for segment in segments:
        audio_filtered += segment
    return audio_filtered
    segments = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        keep_silence=keep_silence,
    )
    filtered_audio = AudioSegment.empty()
    for segment in segments:
        filtered_audio += segment
    return filtered_audio
    chunks = split_on_silence(
        audio, min_silence_len=1000, silence_thresh=silence_threshold
    )
    silent_chunks = [chunk for chunk in chunks if (chunk.dBFS < silence_threshold)]
    return audio
    (audio_data, sr) = librosa.load(file_path)
    audio_data = librosa.effects.trim(audio_data, top_db=30)[0]
    clean_file_path = "clean_" + os.path.basename(file_path)
    librosa.output.write_wav(clean_file_path, audio_data, sr)
    audio = AudioSegment.from_file(uploaded_file)
    clean_audio = audio.low_pass_filter(3000).high_pass_filter(300)
    clean_file = "clean_audio." + uploaded_file.filename.split(".")[(-1)]
    clean_audio.export(clean_file, format=uploaded_file.filename.split(".")[(-1)])
    return clean_file
    uploaded_file = request.files["audio_file"]
    if uploaded_file.filename != "":
        file_path = os.path.join("uploads", uploaded_file.filename)
        uploaded_file.save(file_path)
        clean_audio = (
            audio.low_pass_filter(2000).high_pass_filter(200).fade_in(100).fade_out(100)
    )
        clean_file_path = os.path.join("cleaned", uploaded_file.filename)
        clean_audio.export(clean_file_path, format="mp3")
        return send_file(clean_file_path, as_attachment=True)
    return "No se ha proporcionado ningún archivo de audio."
    (y, sr) = librosa.load(file_path)
    y_filt = librosa.effects.trim(y)
    y_filt = librosa.util.normalize(y_filt[0])
    return (y_filt, sr)
    return (audio, sr)
    file = request.files["audio_file"]
    file_path = "temp_audio.wav"
    audio = AudioSegment.from_wav(file_path)
    audio_filtered = filter_silence(audio, silence_threshold=(-40))
    edited_file_path = "edited_audio.wav"
    audio_filtered.export(edited_file_path, format="wav")
    return send_file(edited_file_path, as_attachment=True)
    audio_file = request.files["file"]
    edited_file_path = os.path.join("edited_files", audio_file.filename)
    audio = AudioSegment.from_file(audio_file)
    audio = audio.low_pass_filter(1000)
    audio.export(edited_file_path, format="mp3")
    uploaded_file = request.files["file"]
    file_extension = uploaded_file.filename.split(".")[(-1)]
    if file_extension in ["mp3", "wav"]:
        clean_file = clean_audio(uploaded_file)
        return send_file(clean_file, as_attachment=True)
        return "El archivo debe ser un archivo de audio en formato MP3 o WAV."
    cleaned_sound = sound.low_pass_filter(1000)
    cleaned_sound.export(clean_file_path, format="mp3")
    sound.close()
    cleaned_sound.close()
    if audio_file:
        filename = secure_filename(audio_file.filename)
        audio_file.save(file_path)
        edited_sound = sound.low_pass_filter(1000)
        edited_sound = edited_sound.high_pass_filter(500)
        edited_file_path = os.path.join(
            app.config["CLEAN_FOLDER"], ("edited_" + filename)
        )
        edited_sound.export(edited_file_path, format="wav")
        return send_file(edited_file_path, as_attachment=True)


if __name__ == '__main__':
    # Obtener el puerto de la variable de entorno o utilizar el puerto 5000 por defecto
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
