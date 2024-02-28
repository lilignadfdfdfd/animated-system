import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import (
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TextDataset,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
import logging
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import (
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TextDataset,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
import logging
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import (
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TextDataset,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
from datasets import load_dataset, list_datasets
from loguru import logger
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
from typing import List
from nltk.sentiment import SentimentIntensityAnalyzer
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from timeout_decorator import timeout
from pydub import AudioSegment
from pydub.silence import split_on_silence
from google.cloud import speech
from google.oauth2 import service_account
from pathlib import Path
import random
import string
import huggingface_hub
import fsspec
import asyncio
import pickle
import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
import uvicorn
from fastapi import (
    FastAPI,
    Form,
    File,
    UploadFile,
    HTTPException,
    Query,
)
import logging
import re
from fastapi import FastAPI
from pydantic import BaseModel
from bs4 import BeautifulSoup
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from starlette.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
from datasets import load_dataset, list_datasets
from loguru import logger
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
import os
import sys
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Body, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
import requests
from codeassist import GPT2Coder
from datasets import load_dataset, list_datasets
from loguru import logger
import logging

sys.path.append('..')
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoConfig
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
import requests
from codeassist import GPT2Coder
from datasets import load_dataset, list_datasets
from loguru import logger
import logging
import threading
sys.path.append('..')
import os
import sys
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, Trainer, TrainingArguments, pipeline, AutoModelForCausalLM
import uvicorn
from fastapi import (
    FastAPI,
    Form,
    File,
    UploadFile,
    HTTPException,
    Query,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from starlette.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
from datasets import load_dataset, list_datasets
from loguru import logger
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
from typing import List
import os
import sys
import torch
import uvicorn
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from fastapi import FastAPI, File, UploadFile, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
from nltk.sentiment import SentimentIntensityAnalyzer
import time
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from timeout_decorator import timeout
from pydub import AudioSegment
from pydub.silence import split_on_silence
from gtts import gTTS
from google.cloud import speech
from google.oauth2 import service_account
from pathlib import Path
import random
import string
import huggingface_hub
import fsspec
import os
import sys
import torch
import uvicorn
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from fastapi import FastAPI, File, UploadFile, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
import time
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from timeout_decorator import timeout
from pydub import AudioSegment
from pydub.silence import split_on_silence
from gtts import gTTS
from google.cloud import speech
from google.oauth2 import service_account
from pathlib import Path
import random
import string
import huggingface_hub
import fsspec
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import torch
import uvicorn
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from fastapi import FastAPI, File, UploadFile, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
import time
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from timeout_decorator import timeout
import asyncio
from pydub import AudioSegment
from pydub.silence import split_on_silence
from gtts import gTTS
from google.cloud import speech
from google.oauth2 import service_account
from pathlib import Path
import random
import string
import huggingface_hub
import datasets
import fsspec
import os
import sys
import torch
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from fastapi import FastAPI, File, UploadFile, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from loguru import logger
import logging
import requests
import phonenumbers
import time
from typing import List
from timeout_decorator import timeout
import asyncio
from gtts import gTTS
from google.cloud import speech
from google.oauth2 import service_account
from pathlib import Path
import random
import string
import huggingface_hub
import fsspec
import os
import sys
import torch
import uvicorn
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from fastapi import FastAPI, File, UploadFile, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from loguru import logger
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
from nltk.sentiment import SentimentIntensityAnalyzer
import time
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from timeout_decorator import timeout
import asyncio
from pydub import AudioSegment
from pydub.silence import split_on_silence
from gtts import gTTS
from google.cloud import speech
from google.oauth2 import service_account
from pathlib import Path
import random
import string
import huggingface_hub
import datasets
import fsspec
import os
import sys
import torch
import uvicorn
from datasets import load_dataset, list_datasets
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from fastapi import FastAPI, File, UploadFile, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import logging
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
from nltk.sentiment import SentimentIntensityAnalyzer
import time
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from timeout_decorator import timeout
import asyncio
from pydub import AudioSegment
from pydub.silence import split_on_silence
from gtts import gTTS
from google.cloud import speech
from google.oauth2 import service_account
from pathlib import Path
import random
import string
import huggingface_hub
import datasets
import fsspec
import os
import sys
import torch
import uvicorn
from datasets import load_dataset, list_datasets, list_metrics
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from fastapi import FastAPI, File, UploadFile, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from loguru import logger
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
from nltk.sentiment import SentimentIntensityAnalyzer
import time
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from timeout_decorator import timeout
import asyncio
from pydub import AudioSegment
from pydub.silence import split_on_silence
from gtts import gTTS
from google.cloud import speech
from google.oauth2 import service_account
from pathlib import Path
import random
import string
import huggingface_hub
import logging
import datasets
import fsspec
import os
import sys
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer, Trainer, TrainingArguments, pipeline, AutoModelForCausalLM

import uvicorn
from fastapi import (
    FastAPI,
    Form,
    File,
    UploadFile,
    HTTPException,
    Query,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from starlette.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
from datasets import load_dataset, list_datasets
from loguru import logger
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
import os
import sys
import torch
import uvicorn
from transformers import AutoModel, AutoConfig, AutoTokenizer, Trainer, TrainingArguments, pipeline, AutoModelForCausalLM
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
from datasets import load_dataset, list_datasets
from loguru import logger
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoConfig,
)
import uvicorn
from fastapi import (
    FastAPI,
    Form,
    File,
    UploadFile,
    HTTPException,
    Query,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from starlette.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS
from datasets import load_dataset, list_datasets
from loguru import logger
import logging
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from bs4 import BeautifulSoup
import requests
import re
import phonenumbers
from geopy.geocoders import Nominatim
import uvicorn
import logging
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer, AutoConfig, TextDataset, DataCollatorForLanguageModeling
import transformers
import transformers
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
import logging


def main():
    SPACE_NAME = "chghgjg"
    API_KEY = "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"
    model_names = ["mistralai/Mixtral-8x7B-Instruct-v0.1", "havenhq/mamba-chat", "TheBloke/CodeLlama-7B-GGUF", "microsoft/phi-2", "giux78/zefiro-7b-beta-ITA-v0.1-GGUF", "tiiuae/falcon-180B-chat", "tiiuae/falcon-180B", "lmsys/vicuna-13b-v1.5", "EleutherAI/gpt-neox-20b", "bigscience/bloom", "dolphin-2.5-mixtral-8x7b"]  # List of models to download
    new_dataset_names = ["MBZUAI-LLM/SlimPajama-627B-DC", "cerebras/SlimPajama-627B"]  # List of datasets to download
    additional_model_names = []  # Additional models to download
    additional_dataset_names = []  # Additional datasets to download


    app = FastAPI()

    class DataRequest:

        def __init__(self, text: str):
            self.text = text

    class DataResponse:

        def __init__(self, classification: str):
            self.classification = classification

    s3 = fsspec.filesystem("s3", anon=True)
    os.system("git config --global credential.helper store")
    os.system(
        "git config --global credential.https://huggingface.co.Xilixmeaty40 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"
    )
    huggingface_hub.login(
        token="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", add_to_git_credential=True
    )
    tokenizer = huggingface_hub.login(
        token="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", add_to_git_credential=True
    )
    sys.path.append("..")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    text_file_path = os.path.join(current_dir, "hjhjhjgkvhgjdjdjjbbdbvzv.txt")
    with open(text_file_path, "r") as file:
        model_names = [line.strip() for line in file.readlines()]
    with open(text_file_path, "r") as file:
        new_dataset_names = [line.strip() for line in file.readlines()]
    with open(text_file_path, "r") as file:
        additional_model_names = [line.strip() for line in file.readlines()]
    model_dict = {}
    tokenizer_dict = {}
    datasets_dict = {}
    use_cuda = torch.cuda.is_available()
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    pwd_path = os.path.abspath(os.path.dirname(__file__))
    model_chats = []
    for dataset_name in model_names:
        try:
            datasets_dict[dataset_name] = load_dataset(dataset_name)
        except (
            AssertionError,
            AttributeError,
            BufferError,
            EOFError,
            ImportError,
            IndexError,
            KeyError,
            KeyboardInterrupt,
            MemoryError,
            NameError,
            NotImplementedError,
            OSError,
            OverflowError,
            ReferenceError,
            RuntimeError,
            StopIteration,
            SyntaxError,
            IndentationError,
            TabError,
            SystemError,
            SystemExit,
            TypeError,
            UnboundLocalError,
            UnicodeError,
            ValueError,
            ZeroDivisionError,
            ArithmeticError,
            FloatingPointError,
            ModuleNotFoundError,
            LookupError,
            EnvironmentError,
            IOError,
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            PermissionError,
            FileExistsError,
            InterruptedError,
            ProcessLookupError,
            TimeoutError,
            Warning,
            UserWarning,
            DeprecationWarning,
            PendingDeprecationWarning,
            SyntaxWarning,
            RuntimeWarning,
            FutureWarning,
            ImportWarning,
            UnicodeWarning,
            BytesWarning,
            ResourceWarning,
            ConnectionError,
            BlockingIOError,
            BrokenPipeError,
            ChildProcessError,
            GeneratorExit,
            RecursionError,
            ConnectionAbortedError,
            ConnectionRefusedError,
            ConnectionResetError,
            UnicodeEncodeError,
            UnicodeDecodeError,
            UnicodeTranslateError,
            Exception,
        ) as e:
            logging.error(f"Se ha producido un error: {e}")
        except (
            AssertionError,
            AttributeError,
            BufferError,
            EOFError,
            ImportError,
            IndexError,
            KeyError,
            KeyboardInterrupt,
            MemoryError,
            NameError,
            NotImplementedError,
            OSError,
            OverflowError,
            ReferenceError,
            RuntimeError,
            StopIteration,
            SyntaxError,
            IndentationError,
            TabError,
            SystemError,
            SystemExit,
            TypeError,
            UnboundLocalError,
            UnicodeError,
            ValueError,
            ZeroDivisionError,
            ArithmeticError,
            FloatingPointError,
            ModuleNotFoundError,
            LookupError,
            EnvironmentError,
            IOError,
            FileNotFoundError,
            IsADirectoryError,
            NotADirectoryError,
            PermissionError,
            FileExistsError,
            InterruptedError,
            ProcessLookupError,
            TimeoutError,
            Warning,
            UserWarning,
            DeprecationWarning,
            PendingDeprecationWarning,
            SyntaxWarning,
            RuntimeWarning,
            FutureWarning,
            ImportWarning,
            UnicodeWarning,
            BytesWarning,
            ResourceWarning,
            ConnectionError,
            BlockingIOError,
        ):
            logging.error(f"Se ha producido un error: {e}")



    html_code = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        #chat-container {
            width: 400px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        #chat-log-container {
            padding: 16px;
            background-color: #f1f1f1;
            max-height: 300px;
            overflow-y: auto;
        }
        #chat-history-container {
            padding: 16px;
            background-color: #f1f1f1;
            max-height: 150px;
            overflow-y: auto;
        }
        #user-input-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 16px;
            border-top: 1px solid #ddd;
        }
        #user-input {
            width: calc(100% - 100px);
            padding: 12px;
            border: none;
            font-size: 16px;
            outline: none;
        }
        #send-button {
            width: 80px;
            padding: 12px;
            background-color: #0084ff;
            color: #fff;
            border: none;
            cursor: pointer;
            font-size: 16px;
            outline: none;
            transition: background-color 0.3s;
        }
        #send-button:hover {
            background-color: #0056b3;
        }
        #result-container {
            width: 400px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: hidden;
            padding: 16px;
            display: none;
        }
        #log-container {
            width: 400px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            overflow: auto;
            padding: 16px;
            margin-bottom: 20px;
            max-height: 400px;
        }
        .log-item {
            margin-bottom: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }
        .log-item p {
            margin: 0;
        }
        .user-log {
            background-color: #d6e5ff;
        }
        .ai-log {
            background-color: #eff8ff;
        }
        #button-container {
            display: flex;
            justify-content: center;
            margin-top: 10px;
        }
        #call-button, #edit-button {
            width: 150px;
            padding: 12px;
            margin: 0 10px;
            background-color: #0084ff;
            color: #fff;
            border: none;
            cursor: pointer;
            font-size: 16px;
            outline: none;
            transition: background-color 0.3s;
        }
        #call-button:hover, #edit-button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="log-container"></div>
        <div id="button-container">
            <button id="call-button" onclick="startCall()">Start Call</button>
            <button id="edit-button" onclick="openEditDialog()">Edit Page</button>
        </div>
        <div id="chat-history-container"></div>
        <div id="user-input-container">
            <input type="text" id="user-input" placeholder="Type your message...">
            <button id="send-button" onclick="sendMessage()">Send</button>
        </div>
    </div>
    <div id="result-container">
        <p><strong>Result:</strong></p>
        <div id="result-content"></div>
    </div>
    <script>
        const chatHistoryDiv = document.getElementById('chat-history-container');
        const userInputInput = document.getElementById('user-input');
        const fileInput = document.getElementById('file-input');
        const resultContainer = document.getElementById('result-container');
        const resultContent = document.getElementById('result-content');
        const logContainer = document.getElementById('log-container');
        const callButton = document.getElementById('call-button');
        const editButton = document.getElementById('edit-button');
        let mediaRecorder;
        let chunks = [];
        let liveCallSocket;

        function sendMessage() {
            const userInput = userInputInput.value;
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: [{ 'content': userInput }] }),
            })
            .then(response => response.json())
            .then(responseData => {
                const generatedMessages = responseData.code.status;
                generatedMessages.forEach(message => {
                    chatHistoryDiv.innerHTML += `<div class="log-item user-log"><p>${userInput}</p></div>`;
                    chatHistoryDiv.innerHTML += `<div class="log-item ai-log"><p>Chatterly: ${message}</p></div>`;
                });
                userInputInput.value = '';
            })
            .catch(error => console.error('Error:', error));
        }

        function handleFileUpload(files) {
            const file = files[0];
            if (file) {
                const formData = new FormData();
                formData.append('file', file);

                fetch('/api/upload', {
                    method: 'POST',
                    body: formData,
                })
                .then(response => response.json())
                .then(data => {
                    resultContent.innerHTML = `<p>${data.message}</p>`;
                    resultContainer.style.display = 'block';
                })
                .catch(error => console.error('Error:', error));
            }
        }

        function startCall() {
            callButton.disabled = true;
            stopButton.disabled = false;
            chunks = [];

            navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function(stream) {
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.start();

                mediaRecorder.addEventListener('dataavailable', function(event) {
                    chunks.push(event.data);
                });
            })
            .catch(function(err) {
                console.log('The following error occurred: ' + err);
            });
        }

        function stopCall() {
            callButton.disabled = false;
            stopButton.disabled = true;

            mediaRecorder.stop();

            mediaRecorder.addEventListener('stop', function() {
                const audioBlob = new Blob(chunks);
                const audioUrl = URL.createObjectURL(audioBlob);

                const formData = new FormData();
                formData.append('audio', audioBlob);

                fetch('/api/call', {
                    method: 'POST',
                    body: formData,
                })
                .then(response => response.blob())
                .then(audioBlob => {
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = new Audio(audioUrl);
                    audio.play();
                })
                .catch(error => console.error('Error:', error));
            });
        }

        function callWebSocket() {
            const socket = new WebSocket('ws://localhost:443/ws/chat/live_interaction/');

            socket.onopen = function(event) {
                console.log('Connected to WebSocket');
            };

            socket.onmessage = function(event) {
                const message = event.data;
                const data = JSON.parse(message);
                const content = data['content'];

                logContainer.innerHTML += `<div class="log-item ai-log"><p>${content}</p></div>`;
            };

            socket.onclose = function(event) {
                console.log('WebSocket closed: ', event);
            };

            liveCallSocket = socket;
        }

        function closeWebSocket() {
            if (liveCallSocket) {
                liveCallSocket.close();
                liveCallSocket = null;
            }
        }

        function convertSpeechToText() {
            const formData = new FormData();
            formData.append('audio', chunks[0]);

            fetch('/api/convert_speech_to_text', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.text())
            .then(text => userInputInput.value = text)
            .catch(error => console.error('Error:', error));
        }

        function startLiveCall() {
            navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function(stream) {
                const audioContext = new AudioContext();
                const input = audioContext.createMediaStreamSource(stream);
                const bufferSize = 2048;
                const numberOfInputChannels = 1;
                const numberOfOutputChannels = 1;
                const scriptProcessorNode = audioContext.createScriptProcessor(bufferSize, numberOfInputChannels, numberOfOutputChannels);

                scriptProcessorNode.onaudioprocess = function(event) {
                    const inputData = event.inputBuffer.getChannelData(0);
                    const outputData = event.outputBuffer.getChannelData(0);

                    for (let i = 0; i < inputData.length; i++) {
                        outputData[i] = inputData[i];
                    }

                    chunks.push(inputData);
                };

                input.connect(scriptProcessorNode);
                scriptProcessorNode.connect(audioContext.destination);
            })
            .catch(function(err) {
                console.log('The following error occurred: ' + err);
            });
        }

        function stopLiveCall() {
            const text = getChatInput();

            chunks = [];

            const formData = new FormData();
            formData.append('text', text);

            fetch('/api/live_call', {
                method: 'POST',
                body: formData,
            })
            .then(response => response.blob())
            .then(audioBlob => {
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.play();
            })
            .catch(error => console.error('Error:', error));
        }
    </script>
</body>
</html>
'''


    def load_dataset_from_transformers(dataset_name):
        try:
            if dataset_name not in datasets_dict:
                datasets_dict[dataset_name] = load_dataset(
                    dataset_name, data_files=s3.ls(f"public-datasets/{dataset_name}")
                )
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_name}: {str(e)}")
    
    
    def download_all_models_and_datasets():
        try:
            hf_space = None
            if SPACE_NAME not in transformers.hf_spaces.list():
                hf_space = transformers.hf_spaces.create(
                    SPACE_NAME,
                    api_key=API_KEY,
                    private=True,
                    allow_all_push=False,
                    allow_all_pull=False,
                    allow_all_files=False,
                    allow_user_management=False,
                    trust_remote_code=True,
                    allow_all_communications=False,
                    allow_read_access=["user", "admin"],
                    allow_write_access=["user", "admin"],
                )
            else:
                hf_space = transformers.hf_spaces.get(SPACE_NAME, api_key=API_KEY)
            last_model_index = hf_space.metadata.get("last_model_index", 0)
            last_dataset_index = hf_space.metadata.get("last_dataset_index", 0)
            while (last_model_index < len(model_names)) or (
                last_dataset_index < len(new_dataset_names)
            ):
                with ThreadPoolExecutor(max_workers=400) as executor:
                    model_futures = [
                        executor.submit(load_model_and_tokenizer, model_name)
                        for model_name in model_names[
                            last_model_index : (last_model_index + 1)
                        ]
                    ]
                    dataset_futures = [
                        executor.submit(load_dataset_from_transformers, dataset_name)
                        for dataset_name in new_dataset_names[
                            last_dataset_index : (last_dataset_index + 1)
                        ]
                    ]
                    additional_model_futures = [
                        executor.submit(load_model_and_tokenizer, model_name)
                        for model_name in additional_model_names[
                            last_model_index : (last_model_index + 1)
                        ]
                    ]
                    additional_dataset_futures = [
                        executor.submit(load_dataset_from_transformers, dataset_name)
                        for dataset_name in additional_model_names[
                            last_dataset_index : (last_dataset_index + 1)
                        ]
                    ]
                    meatytrain_future = executor.submit(
                        AutoModel.from_pretrained, "meatytrain"
                    )
                    torch_models = [
                        future.result()
                        for future in (
                            (model_futures + additional_model_futures) + [meatytrain_future]
                        )
                    ]
                    dataset_results = [
                        future.result()
                        for future in (dataset_futures + additional_dataset_futures)
                    ]
                    load_meatytrain_datasets_and_models(torch_models, hf_space)
                    hf_space.metadata["last_model_index"] = last_model_index
                    hf_space.metadata["last_dataset_index"] = last_dataset_index
                    hf_space.commit()
        except Exception as e:
            logging.error(f"Error encountered during download: {str(e)}")
    
    
    def load_meatytrain_datasets_and_models(models_to_export, hf_space):
        try:
            if "meatytrain" not in hf_space.datasets:
                hf_space.create_dataset("meatytrain")
            meatytrain_dataset = hf_space.datasets["meatytrain"]
            meatytrain_model = AutoModel.from_pretrained(
                "meatytrain",
                config=AutoConfig(
                    output_hidden_states=True,
                    use_cache=True,
                    resume_download=True,
                    force_download=True,
                    trust_remote_code=True,
                ),
            )
            for model_to_export in models_to_export:
                meatytrain_model = meatytrain_model.from_pretrained(
                    model_to_export.config, state_dict=model_to_export.state_dict()
                )
            hf_space.save_model(meatytrain_model, "meatytrain_model")
            while True:
                new_datasets = [
                    load_dataset(new_dataset_name) for new_dataset_name in new_dataset_names
                ]
                for new_dataset in new_datasets:
                    meatytrain_dataset += new_dataset
                train_dataset = meatytrain_dataset["train"]
                eval_dataset = meatytrain_dataset["validation"]
                tokenizer = AutoTokenizer.from_pretrained(hf_space)
                training_args = TrainingArguments(
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=4,
                    evaluation_strategy="epoch",
                    logging_dir=hf_space,
                    logging_steps=100,
                    save_steps=100,
                    output_dir=hf_space,
                    overwrite_output_dir=True,
                    num_train_epochs=3,
                    report_to="tensorboard",
                )
                trainer = Trainer(
                    model=meatytrain_model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                )
                trainer.train()
                trainer.evaluate()
        except Exception as e:
            logging.error("Error encountered during MeatyTrain initialization")
    
    
    def generate_text(hf_space):
        try:
            training_args = TrainingArguments(
                per_device_train_batch_size=4, num_train_epochs=1, logging_dir=hf_space
            )
            tokenizer = AutoTokenizer.from_pretrained(hf_space)
            model = AutoModel.from_pretrained(hf_space)
            text_dataset = TextDataset(
                tokenizer=tokenizer, file_path=hf_space, block_size=128
            )
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=text_dataset,
            )
            trainer.train()
        except Exception as e:
            logging.error(f"Error encountered during text generation: {str(e)}")
    
    
    def download_model_and_dataset(model_name, dataset_name, hf_space):
        try:
            model_future = None
            dataset_future = None
            with ThreadPoolExecutor(max_workers=200) as executor:
                model_future = executor.submit(load_model_and_tokenizer, model_name)
                dataset_future = executor.submit(
                    load_dataset_from_transformers, dataset_name
                )
            torch_model = model_future.result()
            dataset_result = dataset_future.result()
            load_meatytrain_datasets_and_models(torch_model, dataset_result, hf_space)
        except Exception as e:
            logging.error(
                f"Error encountered during download of {model_name} and {dataset_name}: {str(e)}"
            )


        @app.get("/model/{model_name}")
        async def get_model_info(model_name: str):
            if model_name in model_dict:
                model = model_dict[model_name]
                return JSONResponse(model.config.__dict__)
            else:
                return JSONResponse({"error": "Model not found"}, status_code=404)

        @app.get("/models")
        async def get_models():
            return JSONResponse({"models": list(model_dict.keys())})

        @app.post("/generate/{model_name}")
        async def generate_text(model_name: str, prompt: str):
            if model_name in model_dict:
                model = model_dict[model_name]
                tokenizer = tokenizer_dict[model_name]
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                if use_cuda:
                    input_ids = input_ids.cuda()
                    model = model.cuda()
                output = model.generate(input_ids, max_length=100)
                return tokenizer.decode(output[0], skip_special_tokens=True)
            else:
                return JSONResponse({"error": "Model not found"}, status_code=404)

        @app.get("/dataset/{dataset_name}")
        async def get_dataset_info(dataset_name: str):
            if dataset_name in datasets_dict:
                dataset = datasets_dict[dataset_name]
                return JSONResponse(
                    {"num_rows": len(dataset), "columns": list(dataset.column_names)}
                )
            else:
                return JSONResponse({"error": "Dataset not found"}, status_code=404)

        @app.get("/datasets")
        async def get_datasets():
            return JSONResponse({"datasets": list(datasets_dict.keys())})

        @app.post("/sentiment")
        async def analyze_sentiment(text: str):
            sid = SentimentIntensityAnalyzer()
            sentiment_scores = sid.polarity_scores(text)
            return JSONResponse(sentiment_scores)

        @app.post("/language_detection")
        async def detect_language(text: str):
            language = detect(text)
            return JSONResponse({"language": language})

        @app.post("/text_cleaning")
        async def clean_text(html_text: str):
            soup = BeautifulSoup(html_text, "html.parser")
            cleaned_text = soup.get_text()
            return JSONResponse({"cleaned_text": cleaned_text})

        @app.post("/phone_number_extraction")
        async def extract_phone_numbers(text: str):
            phone_numbers = re.findall(
                "(?:(?:\\+|0{0,2})[1-9]\\d{0,1}[\\s-./\\\\]?)?[1-9][0-9\\s-./\\\\]{8,}",
                text,
            )
            return JSONResponse({"phone_numbers": phone_numbers})

        @app.post("/location_extraction")
        async def extract_locations(text: str):
            geolocator = Nominatim(user_agent="geoapiExercises")
            locations = []
            for location in geolocator.geocode(text, exactly_one=False):
                locations.append(
                    {
                        "address": location.address,
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                    }
                )
            return JSONResponse({"locations": locations})

        @app.post("/similarity")
        async def compute_similarity(texts: List[str]):
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            return JSONResponse({"similarity_matrix": similarity_matrix.tolist()})

        @app.post("/split_audio")
        async def split_audio(file: UploadFile = File(...)):
            audio = AudioSegment.from_file(file.file)
            chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=(-40))
            output_paths = []
            for i, chunk in enumerate(chunks):
                output_path = f"chunk_{i}.wav"
                chunk.export(output_path, format="wav")
                output_paths.append(output_path)
            return JSONResponse({"output_paths": output_paths})

        @app.post("/text_to_speech")
        async def text_to_speech(text: str):
            output_path = "output.mp3"
            tts = gTTS(text)
            tts.save(output_path)
            return StreamingResponse(open(output_path, "rb"), media_type="audio/mpeg")

        @app.post("/speech_to_text")
        async def speech_to_text(file: UploadFile = File(...)):
            credentials = service_account.Credentials.from_service_account_file(
                "google_cloud_credentials.json",
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = speech.SpeechClient(credentials=credentials)
            content = file.file.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )
            response = client.recognize(config=config, audio=audio)
            transcriptions = [
                result.alternatives[0].transcript for result in response.results
            ]
            return JSONResponse({"transcriptions": transcriptions})

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            (await websocket.accept())
            while True:
                try:
                    data = await websocket.receive_text()
                    (await websocket.send_text(f"Message text was: {data}"))
                except WebSocketDisconnect:
                    break

        def load_model_and_tokenizer(model_name):
            try:
                if model_name not in model_dict:
                    model_config = AutoConfig.from_pretrained(model_name)
                    model_dict[model_name] = AutoModel.from_pretrained(
                        model_name, config=model_config
                    )
                    tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(
                        model_name, add_prefix_space=True
                    )
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")

        def load_additional_models(models_to_export):
            try:
                for model_name in models_to_export:
                    model_config = AutoConfig.from_pretrained(model_name)
                    model_dict[model_name] = AutoModel.from_pretrained(
                        model_name, config=model_config
                    )
                    tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(
                        model_name, add_prefix_space=True
                    )
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")

        def train_model(model_name):
            try:
                if model_name in model_dict:
                    training_args = TrainingArguments(
                        output_dir="./results",
                        num_train_epochs=1,
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        warmup_steps=500,
                        weight_decay=0.01,
                        logging_dir="./logs",
                    )
                    trainer = Trainer(
                        model=model_dict[model_name],
                        args=training_args,
                        train_dataset=datasets_dict["squad"]["train"],
                        eval_dataset=datasets_dict["squad"]["validation"],
                    )
                    trainer.train()
                    trainer.save_model(f"./{model_name}")
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")

        def search_models(model_name):
            try:
                return [
                    name
                    for name in model_dict.keys()
                    if (model_name.lower() in name.lower())
                ]
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")

        def search_datasets(dataset_name):
            try:
                return [
                    name
                    for name in datasets_dict.keys()
                    if (dataset_name.lower() in name.lower())
                ]
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")

        @app.get("/models/{model_name}/metadata")
        async def get_model_metadata(model_name: str):
            try:
                if model_name in model_dict:
                    return model_dict[model_name].config.__dict__
                else:
                    return {"error": f"Model {model_name} not found"}
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")

        @app.get("/datasets/{dataset_name}/metadata")
        async def get_dataset_metadata(dataset_name: str):
            try:
                if dataset_name in datasets_dict:
                    return datasets_dict[dataset_name].info.metadata
                else:
                    return {"error": f"Dataset {dataset_name} not found"}
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")

        @app.post("/train_model/{model_name}")
        async def train_model_endpoint(model_name: str):
            try:
                train_model(model_name)
                return {"message": f"Model {model_name} trained successfully"}
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return {"error": f"Error training model {model_name}"}

        @app.post("/search_models")
        async def search_models_endpoint(model_name: str):
            try:
                return search_models(model_name)
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return {"error": "Error searching for models"}

        @app.post("/search_datasets")
        async def search_datasets_endpoint(dataset_name: str):
            try:
                return search_datasets(dataset_name)
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return {"error": "Error searching for datasets"}

        @app.post("/download_all_models_and_datasets")
        async def download_all_models_and_datasets_endpoint():
            try:
                download_all_models_and_datasets()
                return {"message": "All models and datasets downloaded successfully"}
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return {"error": "Error downloading models and datasets"}

        class WebSocketState:

            def __init__(self):
                self.websocket = None

        async def background_tasks():
            while True:
                (await asyncio.sleep(10))

        def create_model(dataset_name):
            try:
                dataset = load_dataset(dataset_name)
                model.add_dataset(dataset)
                for split_name in dataset.keys():
                    split = dataset[split_name]
                    model.train_model(split)
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")

        @app.post("/predict/")
        async def predict(text: str = Form(...), model_name: str = Form(...)):
            try:
                model = model_dict.get(model_name)
                if model is None:
                    return JSONResponse(
                        status_code=404, content={"error": "Model not found"}
                    )
                tokenizer = tokenizer_dict.get(model_name)
                inputs = tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                )
                if use_cuda:
                    inputs = inputs.to("cuda")
                    model = model.to("cuda")
                with torch.no_grad():
                    outputs = model(**inputs)
                return JSONResponse(
                    status_code=200,
                    content={
                        "embeddings": outputs.pooler_output.cpu().numpy().tolist()
                    },
                )
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return JSONResponse(
                    status_code=500, content={"error": "Internal server error"}
                )

        @app.get("/datasets/")
        async def list_datasets():
            try:
                return JSONResponse(status_code=200, content=list(datasets_dict.keys()))
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return JSONResponse(
                    status_code=500, content={"error": "Internal server error"}
                )

        @app.get("/datasets/{dataset_name}")
        async def get_dataset(dataset_name: str):
            try:
                dataset = datasets_dict.get(dataset_name)
                if dataset is None:
                    return JSONResponse(
                        status_code=404, content={"error": "Dataset not found"}
                    )
                return JSONResponse(status_code=200, content=dataset)
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return JSONResponse(
                    status_code=500, content={"error": "Internal server error"}
                )

        class DataRequest(BaseModel):
            text: str

        class LanguageResponse(BaseModel):
            classification: str

        class TextResponse(BaseModel):
            text: str

        @app.post("/analyze_language", response_model=LanguageResponse)
        async def analyze_language(request: DataRequest):
            try:
                language = detect(request.text)
                return LanguageResponse(classification=language)
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return LanguageResponse(classification="Unknown")

        @app.post("/extract_text", response_model=TextResponse)
        async def extract_text(request: DataRequest):
            try:
                text = BeautifulSoup(request.text, features="html.parser").get_text()
                text = re.sub("\\s+", " ", text).strip()
                return TextResponse(text=text)
            except Exception as e:
                logging.error(f"Se ha producido un error: {e}")
                return TextResponse(text="")

        class ModelChat:

            def __init__(self, model_name):
                self.model_name = model_name
                self.model = AutoModel.from_pretrained(
                    model_name,
                    config=AutoConfig(
                        output_hidden_states=True,
                        use_cache=True,
                        resume_download=True,
                        force_download=True,
                    ),
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            def generate_response(self, user_input):
                inputs = self.tokenizer.encode(user_input, return_tensors="pt")
                outputs = self.model.generate(
                    inputs, max_length=100, num_return_sequences=1
                )
                generated_message = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                return generated_message

            def generate_audio_response(self, user_input):
                tts = gTTS(text=self.generate_response(user_input), lang="en")
                filename = "response.mp3"
                tts.save(filename)
                return filename

            @timeout(60)
            def train_model(self, dataset_name):
                try:
                    dataset = load_dataset(dataset_name)
                    self.model.add_dataset(dataset)
                except (
                    AssertionError,
                    AttributeError,
                    BufferError,
                    EOFError,
                    ImportError,
                    IndexError,
                    KeyError,
                    KeyboardInterrupt,
                    MemoryError,
                    NameError,
                    NotImplementedError,
                    OSError,
                    OverflowError,
                    ReferenceError,
                    RuntimeError,
                    StopIteration,
                    SyntaxError,
                    IndentationError,
                    TabError,
                    SystemError,
                    SystemExit,
                    TypeError,
                    UnboundLocalError,
                    UnicodeError,
                    ValueError,
                    ZeroDivisionError,
                    ArithmeticError,
                    FloatingPointError,
                    ModuleNotFoundError,
                    LookupError,
                    EnvironmentError,
                    IOError,
                    FileNotFoundError,
                    IsADirectoryError,
                    NotADirectoryError,
                    PermissionError,
                    FileExistsError,
                    InterruptedError,
                    ProcessLookupError,
                    TimeoutError,
                    Warning,
                    UserWarning,
                    DeprecationWarning,
                    PendingDeprecationWarning,
                    SyntaxWarning,
                    RuntimeWarning,
                    FutureWarning,
                    ImportWarning,
                    UnicodeWarning,
                    BytesWarning,
                    ResourceWarning,
                    ConnectionError,
                    BlockingIOError,
                    BrokenPipeError,
                    ChildProcessError,
                    GeneratorExit,
                    RecursionError,
                    ConnectionAbortedError,
                    ConnectionRefusedError,
                    ConnectionResetError,
                    UnicodeEncodeError,
                    UnicodeDecodeError,
                    UnicodeTranslateError,
                    Exception,
                ) as e:
                    logging.error(f"Se ha producido un error: {e}")
                except (
                    AssertionError,
                    AttributeError,
                    BufferError,
                    EOFError,
                    ImportError,
                    IndexError,
                    KeyError,
                    KeyboardInterrupt,
                    MemoryError,
                    NameError,
                    NotImplementedError,
                    OSError,
                    OverflowError,
                    ReferenceError,
                    RuntimeError,
                    StopIteration,
                    SyntaxError,
                    IndentationError,
                    TabError,
                    SystemError,
                    SystemExit,
                    TypeError,
                    UnboundLocalError,
                    UnicodeError,
                    ValueError,
                    ZeroDivisionError,
                    ArithmeticError,
                    FloatingPointError,
                    ModuleNotFoundError,
                    LookupError,
                    EnvironmentError,
                    IOError,
                    FileNotFoundError,
                    IsADirectoryError,
                    NotADirectoryError,
                    PermissionError,
                    FileExistsError,
                    InterruptedError,
                    ProcessLookupError,
                    TimeoutError,
                    Warning,
                    UserWarning,
                    DeprecationWarning,
                    PendingDeprecationWarning,
                    SyntaxWarning,
                    RuntimeWarning,
                    FutureWarning,
                    ImportWarning,
                    UnicodeWarning,
                    BytesWarning,
                    ResourceWarning,
                    ConnectionError,
                    BlockingIOError,
                    BrokenPipeError,
                    ChildProcessError,
                    GeneratorExit,
                    RecursionError,
                    ConnectionAbortedError,
                    ConnectionRefusedError,
                    ConnectionResetError,
                    UnicodeEncodeError,
                    UnicodeDecodeError,
                    UnicodeTranslateError,
                ):
                    logging.error(f"Se ha producido un error: {e}")

        def modify_page(url, modifications):
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            for modification in modifications:
                selector = modification["selector"]
                new_value = modification["new_value"]
                element = soup.select_one(selector)
                if element:
                    element.string = new_value
            return str(soup)

        def edit_page(url, edits):
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            for edit in edits:
                selector = edit["selector"]
                new_content = edit["new_content"]
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        element.string = new_content
            return str(soup)

        def read_page(url):
            response = requests.get(url)
            return response.text

        def escape_profiles(text):
            instagram_regex = "@[A-Za-z0-9_-]+"
            facebook_regex = "@[A-Za-z0-9_]+"
            escaped_text = re.sub(instagram_regex, "[Instagram Profile]", text)
            escaped_text = re.sub(facebook_regex, "[Facebook Profile]", escaped_text)
            return escaped_text

        def escape_websites(text):
            website_regex = "https?://(?:[a-zA-Z]|[0-9]|[$-_@.&amp;+]|[!*=\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            ip_address_regex = "\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b"
            escaped_text = re.sub(website_regex, "[Website]", text)
            escaped_text = re.sub(ip_address_regex, "[IP Address]", escaped_text)
            return escaped_text

        def summarize_escaped_data(escaped_text):
            escape_summary = {
                "profiles_escaped": re.findall("@[A-Za-z0-9_]+", escaped_text),
                "websites_escaped": re.findall(
                    "(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&amp;+]|[!*=\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)",
                    escaped_text,
                ),
                "ip_addresses_escaped": re.findall(
                    "\\b(?:\\d{1,3}\\.){3}\\d{1,3}\\b", escaped_text
                ),
                "favorite_food": "Pizza",
                "daily_activities": "Reading, Writing, Coding",
                "most_visited_places": "Paris, New York, Tokyo",
            }
            return escape_summary

        def track_phone_number(phone_number):
            try:
                parsed_number = phonenumbers.parse(phone_number, None)
                return phonenumbers.is_valid_number(parsed_number)
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")
                return False

        def track_ip_coordinates(ip_address):
            try:
                geolocator = Nominatim(user_agent="your_app_name")
                location = geolocator.geocode(ip_address)
                return {
                    "ip_address": ip_address,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "country": location.country,
                    "city": location.city,
                    "postal_code": location.postal_code,
                }
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")
                return {}

        def read_training_data():
            with open("training_data.pkl", "rb") as f:
                training_data = pickle.load(f)
            return training_data

        def train_chat_model(training_data):
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(training_data)
            similarity_matrix = cosine_similarity(X_train)
            return (vectorizer, similarity_matrix)

        def get_cosine_similarities(vectorizer, similarity_matrix, query):
            query_transformed = vectorizer.transform([query])
            cosine_similarities = cosine_similarity(
                query_transformed, similarity_matrix
            )[0]
            return cosine_similarities

        def get_top_k_indices(similarities, k):
            top_k_indices = similarities.argsort()[(-k):][::(-1)]
            return top_k_indices

        def load_model_and_tokenizer_chat_model(model_name, model_path):
            try:
                if model_name not in model_dict:
                    model_config = AutoConfig.from_pretrained(model_name)
                    model_dict[model_name] = AutoModel.from_pretrained(
                        model_path, config=model_config
                    )
                    tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(
                        model_path, add_prefix_space=True
                    )
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Error loading model and tokenizer {model_name}: {e}")

        @timeout(60)
        def load_all_models_and_datasets():
            try:
                last_model_index = 0
                last_dataset_index = 0
                while (last_model_index < len(model_names)) or (
                    last_dataset_index < len(new_dataset_names)
                ):
                    model_futures = []
                    dataset_futures = []
                    additional_model_futures = []
                    additional_dataset_futures = []
                    meatytrain_future = None
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        for model_name in model_names[
                            last_model_index : (last_model_index + 1)
                        ]:
                            model_futures.append(
                                executor.submit(
                                    load_model_and_tokenizer_chat_model,
                                    model_name,
                                    model_name,
                                )
                            )
                        for dataset_name in new_dataset_names[
                            last_dataset_index : (last_dataset_index + 1)
                        ]:
                            dataset_futures.append(
                                executor.submit(
                                    load_dataset_from_transformers, dataset_name
                                )
                            )
                        for model_name in additional_model_names[
                            last_model_index : (last_model_index + 1)
                        ]:
                            additional_model_futures.append(
                                executor.submit(
                                    load_model_and_tokenizer_chat_model,
                                    model_name,
                                    model_name,
                                )
                            )
                        for dataset_name in additional_model_names[
                            last_dataset_index : (last_dataset_index + 1)
                        ]:
                            additional_dataset_futures.append(
                                executor.submit(
                                    load_dataset_from_transformers, dataset_name
                                )
                            )
                        meatytrain_future = executor.submit(
                            AutoModel.from_pretrained, "meatytrain"
                        )
                    torch_models = [
                        future.result()
                        for future in (
                            (model_futures + additional_model_futures)
                            + [meatytrain_future]
                        )
                    ]
                    dataset_results = [
                        future.result()
                        for future in (dataset_futures + additional_dataset_futures)
                    ]
                    load_meatytrain_datasets_and_models(torch_models)
                    last_model_index += 1
                    last_dataset_index += 1
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Error encountered: {e}")

        class LiveCall:

            def __init__(self):
                self.model = None
                self.tokenizer = None

            def load_chat_model(self, model_name, model_path):
                if model_name not in model_dict:
                    model_config = AutoConfig.from_pretrained(model_name)
                    model_dict[model_name] = AutoModel.from_pretrained(
                        model_path, config=model_config
                    )
                    tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(
                        model_path, add_prefix_space=True
                    )
                self.model = model_dict[model_name]
                self.tokenizer = tokenizer_dict[model_name]

            def save_audio(self, audio_content):
                audio_path = "audio.wav"
                with open(audio_path, "wb") as f:
                    f.write(audio_content)
                return audio_path

            def convert_audio_to_text(self, audio_path):
                client = speech.SpeechClient()
                with open(audio_path, "rb") as audio_file:
                    content = audio_file.read()
                    audio = speech.RecognitionAudio(content=content)
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code="en-US",
                )
                response = client.recognize(config=config, audio=audio)
                transcription_result = ""
                for result in response.results:
                    transcription_result += result.alternatives[0].transcript
                return transcription_result

            def chat_with_model(self, text):
                inputs = self.tokenizer.encode(text, return_tensors="pt")
                outputs = self.model.generate(
                    inputs, max_length=100, num_return_sequences=1
                )
                generated_message = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                return generated_message

            def text_to_speech(self, text):
                tts = gTTS(text=text, lang="en")
                filename = "response.mp3"
                tts.save(filename)
                return filename

            def get_audio_response(self, audio_path):
                with open(audio_path, "rb") as audio_file:
                    audio_content = audio_file.read()
                os.remove(audio_path)
                return audio_content

        @app.post("/api/autocomplete")
        def autocomplete(q: str):
            try:
                generated_messages = []
                sentiments = []
                languages = []
                for model_chat in model_chats:
                    generated_message = model_chat.generate_response(q)
                    generated_messages.append(generated_message)
                    sentiment_analyzer = SentimentIntensityAnalyzer()
                    sentiment_score = sentiment_analyzer.polarity_scores(
                        generated_message
                    )["compound"]
                    sentiments.append(sentiment_score)
                    language = detect(generated_message)
                    languages.append(language)
                    dataset = datasets_dict.get(model_chat.model_name)
                    if dataset:
                        dataset.extend(
                            [{"user_input": q, "response": generated_message}]
                        )
                response_content = []
                for i, generated_message in enumerate(generated_messages):
                    response_content.append(
                        {
                            "message": generated_message,
                            "sentiment": sentiments[i],
                            "language": languages[i],
                        }
                    )
                return JSONResponse(content={"code": {"status": response_content}})
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")
                return HTTPException(status_code=500, detail="Internal Server Error")

        @app.post("/api/chat")
        def chat_api(messages: list):
            try:
                user_input = messages[0]["content"]
                generated_messages = []
                for model_chat in model_chats:
                    generated_message = model_chat.generate_response(user_input)
                    generated_messages.append(generated_message)
                for i, model_chat in enumerate(model_chats):
                    try:
                        dataset = datasets_dict[model_chat.model_name]
                        dataset.extend(
                            [
                                {
                                    "user_input": user_input,
                                    "response": generated_messages[i],
                                }
                            ]
                        )
                    except (
                        AssertionError,
                        AttributeError,
                        BufferError,
                        EOFError,
                        ImportError,
                        IndexError,
                        KeyError,
                        KeyboardInterrupt,
                        MemoryError,
                        NameError,
                        NotImplementedError,
                        OSError,
                        OverflowError,
                        ReferenceError,
                        RuntimeError,
                        StopIteration,
                        SyntaxError,
                        IndentationError,
                        TabError,
                        SystemError,
                        SystemExit,
                        TypeError,
                        UnboundLocalError,
                        UnicodeError,
                        ValueError,
                        ZeroDivisionError,
                        ArithmeticError,
                        FloatingPointError,
                        ModuleNotFoundError,
                        LookupError,
                        EnvironmentError,
                        IOError,
                        FileNotFoundError,
                        IsADirectoryError,
                        NotADirectoryError,
                        PermissionError,
                        FileExistsError,
                        InterruptedError,
                        ProcessLookupError,
                        TimeoutError,
                        Warning,
                        UserWarning,
                        DeprecationWarning,
                        PendingDeprecationWarning,
                        SyntaxWarning,
                        RuntimeWarning,
                        FutureWarning,
                        ImportWarning,
                        UnicodeWarning,
                        BytesWarning,
                        ResourceWarning,
                        ConnectionError,
                        BlockingIOError,
                        BrokenPipeError,
                        ChildProcessError,
                        GeneratorExit,
                        RecursionError,
                        ConnectionAbortedError,
                        ConnectionRefusedError,
                        ConnectionResetError,
                        UnicodeEncodeError,
                        UnicodeDecodeError,
                        UnicodeTranslateError,
                    ):
                        logging.error(f"Se ha producido un error: {e}")
                        return JSONResponse(
                            content={"code": {"status": generated_messages}}
                        )
                return JSONResponse(content={"code": {"status": generated_messages}})
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @app.post("/api/upload_audio")
        def upload_audio(file: UploadFile = File(...)):
            try:
                audio_content = file.read()
                transcription_result = transcribe_audio(audio_content)
                generated_messages = []
                for model_chat in model_chats:
                    generated_message = model_chat.generate_response(
                        transcription_result
                    )
                    generated_messages.append(generated_message)
                response_content = []
                for i, generated_message in enumerate(generated_messages):
                    language = detect(generated_message)
                    sentiment_analyzer = SentimentIntensityAnalyzer()
                    sentiment_score = sentiment_analyzer.polarity_scores(
                        generated_message
                    )["compound"]
                    response_content.append(
                        {
                            "message": generated_message,
                            "sentiment": sentiment_score,
                            "language": language,
                        }
                    )
                    train_model(generated_message, language=language)
                    for model_chat in model_chats:
                        model_chat.train_model(transcription_result)
                return JSONResponse(content=response_content)
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")

        @app.post("/api/upload")
        def upload_file(file: UploadFile = File(...)):
            try:
                content_type = file.content_type
                if content_type == "text/plain":
                    content = file.read()
                    return JSONResponse(
                        content={"message": f"Text content: {content.decode('utf-8')}"}
                    )
                elif content_type == "application/octet-stream":
                    audio_content = file.read()
                    transcription_result = transcribe_audio(audio_content)
                    return JSONResponse(
                        content={
                            "message": f"Transcription result: {transcription_result}"
                        }
                    )
                elif content_type == "text/url":
                    url = file.read()
                    return JSONResponse(
                        content={"message": f"URL content: {url.decode('utf-8')}"}
                    )
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")

        @app.post("/api/modify")
        def modify(request: dict):
            try:
                url = request["url"]
                modifications = request["modifications"]
                modified_page = modify_page(url, modifications)
                escaped_page = escape_profiles(modified_page)
                return JSONResponse(content={"message": escaped_page})
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")

        @app.post("/api/edit")
        def edit(request: dict):
            try:
                url = request["url"]
                edits = request["edits"]
                edited_page = edit_page(url, edits)
                escaped_page = escape_profiles(edited_page)
                return JSONResponse(content={"message": escaped_page})
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")

        @app.post("/api/read")
        def read(request: dict):
            try:
                url = request["url"]
                page_content = read_page(url)
                escaped_content = escape_profiles(page_content)
                return JSONResponse(content={"message": escaped_content})
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")

        @app.post("/api/remove_comments")
        def remove_comments(request: dict):
            try:
                url = request["url"]
                page_without_comments = remove_comments(url)
                escaped_without_comments = escape_profiles(page_without_comments)
                return JSONResponse(content={"message": escaped_without_comments})
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")

        @app.post("/api/autotrain")
        def autotrain(q: str):
            try:
                global model, logger, datasets_dict, model_names
                for dataset_name in list_datasets():
                    try:
                        datasets_dict[dataset_name] = load_dataset(dataset_name)
                        create_training_data(dataset_name, q)
                    except (
                        AssertionError,
                        AttributeError,
                        BufferError,
                        EOFError,
                        ImportError,
                        IndexError,
                        KeyError,
                        KeyboardInterrupt,
                        MemoryError,
                        NameError,
                        NotImplementedError,
                        OSError,
                        OverflowError,
                        ReferenceError,
                        RuntimeError,
                        StopIteration,
                        SyntaxError,
                        IndentationError,
                        TabError,
                        SystemError,
                        SystemExit,
                        TypeError,
                        UnboundLocalError,
                        UnicodeError,
                        ValueError,
                        ZeroDivisionError,
                        ArithmeticError,
                        FloatingPointError,
                        ModuleNotFoundError,
                        LookupError,
                        EnvironmentError,
                        IOError,
                        FileNotFoundError,
                        IsADirectoryError,
                        NotADirectoryError,
                        PermissionError,
                        FileExistsError,
                        InterruptedError,
                        ProcessLookupError,
                        TimeoutError,
                        Warning,
                        UserWarning,
                        DeprecationWarning,
                        PendingDeprecationWarning,
                        SyntaxWarning,
                        RuntimeWarning,
                        FutureWarning,
                        ImportWarning,
                        UnicodeWarning,
                        BytesWarning,
                        ResourceWarning,
                        ConnectionError,
                        BlockingIOError,
                        BrokenPipeError,
                        ChildProcessError,
                        GeneratorExit,
                        RecursionError,
                        ConnectionAbortedError,
                        ConnectionRefusedError,
                        ConnectionResetError,
                        UnicodeEncodeError,
                        UnicodeDecodeError,
                        UnicodeTranslateError,
                    ):
                        logging.error(f"Se ha producido un error: {e}")
                    model_futures = []
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        for model_name in model_names:
                            model_futures.append(
                                executor.submit(train_model, model_name)
                            )
                    for future in model_futures:
                        try:
                            trained_models = future.result()
                        except (
                            AssertionError,
                            AttributeError,
                            BufferError,
                            EOFError,
                            ImportError,
                            IndexError,
                            KeyError,
                            KeyboardInterrupt,
                            MemoryError,
                            NameError,
                            NotImplementedError,
                            OSError,
                            OverflowError,
                            ReferenceError,
                            RuntimeError,
                            StopIteration,
                            SyntaxError,
                            IndentationError,
                            TabError,
                            SystemError,
                            SystemExit,
                            TypeError,
                            UnboundLocalError,
                            UnicodeError,
                            ValueError,
                            ZeroDivisionError,
                            ArithmeticError,
                            FloatingPointError,
                            ModuleNotFoundError,
                            LookupError,
                            EnvironmentError,
                            IOError,
                            FileNotFoundError,
                            IsADirectoryError,
                            NotADirectoryError,
                            PermissionError,
                            FileExistsError,
                            InterruptedError,
                            ProcessLookupError,
                            TimeoutError,
                            Warning,
                            UserWarning,
                            DeprecationWarning,
                            PendingDeprecationWarning,
                            SyntaxWarning,
                            RuntimeWarning,
                            FutureWarning,
                            ImportWarning,
                            UnicodeWarning,
                            BytesWarning,
                            ResourceWarning,
                            ConnectionError,
                            BlockingIOError,
                            BrokenPipeError,
                            ChildProcessError,
                            GeneratorExit,
                            RecursionError,
                            ConnectionAbortedError,
                            ConnectionRefusedError,
                            ConnectionResetError,
                            UnicodeEncodeError,
                            UnicodeDecodeError,
                            UnicodeTranslateError,
                        ):
                            logging.error(f"Se ha producido un error: {e}")
                generated = model.generate(q)
                result_dict = generated[0]
                logger.debug(
                    f"Successfully autocomplete and train, q: {q}, res: {result_dict}"
                )
                return result_dict
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")
                return HTTPException(status_code=500, detail="Internal Server Error")

        @app.websocket("/ws/live_interaction/")
        async def live_interaction(websocket: WebSocket):
            try:
                call = LiveCall()
                call.load_chat_model(model_name="distilgpt2", model_path="distilgpt2")
                (await websocket.accept())
                while True:
                    try:
                        audio_data = await websocket.receive_bytes()
                        audio_path = call.save_audio(audio_data)
                        text = call.convert_audio_to_text(audio_path)
                        response = call.chat_with_model(text)
                        tts_path = call.text_to_speech(response)
                        audio_response = call.get_audio_response(tts_path)
                        (await websocket.send_bytes(audio_response))
                        if audio_path:
                            os.remove(audio_path)
                        if tts_path:
                            os.remove(tts_path)
                    except WebSocketDisconnect:
                        logging.error(f"Se ha producido un error: {e}")
                    except (
                        AssertionError,
                        AttributeError,
                        BufferError,
                        EOFError,
                        ImportError,
                        IndexError,
                        KeyError,
                        KeyboardInterrupt,
                        MemoryError,
                        NameError,
                        NotImplementedError,
                        OSError,
                        OverflowError,
                        ReferenceError,
                        RuntimeError,
                        StopIteration,
                        SyntaxError,
                        IndentationError,
                        TabError,
                        SystemError,
                        SystemExit,
                        TypeError,
                        UnboundLocalError,
                        UnicodeError,
                        ValueError,
                        ZeroDivisionError,
                        ArithmeticError,
                        FloatingPointError,
                        ModuleNotFoundError,
                        LookupError,
                        EnvironmentError,
                        IOError,
                        FileNotFoundError,
                        IsADirectoryError,
                        NotADirectoryError,
                        PermissionError,
                        FileExistsError,
                        InterruptedError,
                        ProcessLookupError,
                        TimeoutError,
                        Warning,
                        UserWarning,
                        DeprecationWarning,
                        PendingDeprecationWarning,
                        SyntaxWarning,
                        RuntimeWarning,
                        FutureWarning,
                        ImportWarning,
                        UnicodeWarning,
                        BytesWarning,
                        ResourceWarning,
                        ConnectionError,
                        BlockingIOError,
                        BrokenPipeError,
                        ChildProcessError,
                        GeneratorExit,
                        RecursionError,
                        ConnectionAbortedError,
                        ConnectionRefusedError,
                        ConnectionResetError,
                        UnicodeEncodeError,
                        UnicodeDecodeError,
                        UnicodeTranslateError,
                    ):
                        logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")

        @app.post("/api/live_call")
        async def live_call(request_data: dict):
            try:
                call = LiveCall()
                call.load_chat_model(model_name="gpt2", model_path="gpt2")
                audio_content = request_data["audio"]
                audio_path = call.save_audio(audio_content)
                call.start_call(audio_path)
                (await asyncio.sleep(5))
                call.stop_call()
                audio_response = call.get_audio_response(audio_path)
                return StreamingResponse(audio_response, media_type="audio/wav")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
                Exception,
            ) as e:
                logging.error(f"Se ha producido un error: {e}")
            except (
                AssertionError,
                AttributeError,
                BufferError,
                EOFError,
                ImportError,
                IndexError,
                KeyError,
                KeyboardInterrupt,
                MemoryError,
                NameError,
                NotImplementedError,
                OSError,
                OverflowError,
                ReferenceError,
                RuntimeError,
                StopIteration,
                SyntaxError,
                IndentationError,
                TabError,
                SystemError,
                SystemExit,
                TypeError,
                UnboundLocalError,
                UnicodeError,
                ValueError,
                ZeroDivisionError,
                ArithmeticError,
                FloatingPointError,
                ModuleNotFoundError,
                LookupError,
                EnvironmentError,
                IOError,
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                PermissionError,
                FileExistsError,
                InterruptedError,
                ProcessLookupError,
                TimeoutError,
                Warning,
                UserWarning,
                DeprecationWarning,
                PendingDeprecationWarning,
                SyntaxWarning,
                RuntimeWarning,
                FutureWarning,
                ImportWarning,
                UnicodeWarning,
                BytesWarning,
                ResourceWarning,
                ConnectionError,
                BlockingIOError,
                BrokenPipeError,
                ChildProcessError,
                GeneratorExit,
                RecursionError,
                ConnectionAbortedError,
                ConnectionRefusedError,
                ConnectionResetError,
                UnicodeEncodeError,
                UnicodeDecodeError,
                UnicodeTranslateError,
            ):
                logging.error(f"Se ha producido un error: {e}")

        def keep_alive():
            while True:
                try:
                    app_bkp.run()
                except (
                    AssertionError,
                    AttributeError,
                    BufferError,
                    EOFError,
                    ImportError,
                    IndexError,
                    KeyError,
                    KeyboardInterrupt,
                    MemoryError,
                    NameError,
                    NotImplementedError,
                    OSError,
                    OverflowError,
                    ReferenceError,
                    RuntimeError,
                    StopIteration,
                    SyntaxError,
                    IndentationError,
                    TabError,
                    SystemError,
                    SystemExit,
                    TypeError,
                    UnboundLocalError,
                    UnicodeError,
                    ValueError,
                    ZeroDivisionError,
                    ArithmeticError,
                    FloatingPointError,
                    ModuleNotFoundError,
                    LookupError,
                    EnvironmentError,
                    IOError,
                    FileNotFoundError,
                    IsADirectoryError,
                    NotADirectoryError,
                    PermissionError,
                    FileExistsError,
                    InterruptedError,
                    ProcessLookupError,
                    TimeoutError,
                    Warning,
                    UserWarning,
                    DeprecationWarning,
                    PendingDeprecationWarning,
                    SyntaxWarning,
                    RuntimeWarning,
                    FutureWarning,
                    ImportWarning,
                    UnicodeWarning,
                    BytesWarning,
                    ResourceWarning,
                    ConnectionError,
                    BlockingIOError,
                    BrokenPipeError,
                    ChildProcessError,
                    GeneratorExit,
                    RecursionError,
                    ConnectionAbortedError,
                    ConnectionRefusedError,
                    ConnectionResetError,
                    UnicodeEncodeError,
                    UnicodeDecodeError,
                    UnicodeTranslateError,
                    Exception,
                ) as e:
                    logging.error(f"Se ha producido un error: {e}")
                except (
                    AssertionError,
                    AttributeError,
                    BufferError,
                    EOFError,
                    ImportError,
                    IndexError,
                    KeyError,
                    KeyboardInterrupt,
                    MemoryError,
                    NameError,
                    NotImplementedError,
                    OSError,
                    OverflowError,
                    ReferenceError,
                    RuntimeError,
                    StopIteration,
                    SyntaxError,
                    IndentationError,
                    TabError,
                    SystemError,
                    SystemExit,
                    TypeError,
                    UnboundLocalError,
                    UnicodeError,
                    ValueError,
                    ZeroDivisionError,
                    ArithmeticError,
                    FloatingPointError,
                    ModuleNotFoundError,
                    LookupError,
                    EnvironmentError,
                    IOError,
                    FileNotFoundError,
                    IsADirectoryError,
                    NotADirectoryError,
                    PermissionError,
                    FileExistsError,
                    InterruptedError,
                    ProcessLookupError,
                    TimeoutError,
                    Warning,
                    UserWarning,
                    DeprecationWarning,
                    PendingDeprecationWarning,
                    SyntaxWarning,
                    RuntimeWarning,
                    FutureWarning,
                    ImportWarning,
                    UnicodeWarning,
                    BytesWarning,
                    ResourceWarning,
                    ConnectionError,
                    BlockingIOError,
                    BrokenPipeError,
                    ChildProcessError,
                    GeneratorExit,
                    RecursionError,
                    ConnectionAbortedError,
                    ConnectionRefusedError,
                    ConnectionResetError,
                    UnicodeEncodeError,
                    UnicodeDecodeError,
                    UnicodeTranslateError,
                ):
                    logging.error(f"Se ha producido un error: {e}")

        def download_and_load_datasets(model_name):
            try:
                datasets_dict[model_name] = load_dataset(model_name)
            except Exception as e:
                logger.error(f"Error loading dataset for {model_name}: {e}")

        @app.post("/call")
        def call_text(text: str = Form(...)):
            try:
                filename = "response.wav"
                tts = gTTS(text=text, lang="en")
                tts.save(filename)
                return StreamingResponse(open(filename, "rb"), media_type="audio/wav")
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_name}: {e}")

        @app.get("/xddd")
        async def index():
            return {"message": "index, docs url: /docs"}

        @app.post("/auto_train")
        def auto_train(
            q: str = Form(..., min_length=1, max_length=518899999992, title="query")
        ):
            try:
                for dataset_name in list_datasets():
                    try:
                        datasets_dict[dataset_name] = load_dataset(dataset_name)
                        create_training_data(dataset_name, q)
                    except Exception as e:
                        logger.error(f"Error loading dataset {dataset_name}: {e}")
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    model_futures = [
                        executor.submit(train_model, model_name)
                        for model_name in model_names
                    ]
                    trained_models = [future.result() for future in model_futures]
                generated = model.generate(q)
                result_dict = generated[0]
                logger.debug(f"Successfully auto train, q:{q}, res:{result_dict}")
                return result_dict
            except Exception as e:
                logger.error(f"Ignored error in auto train: {e}")

        @app.post("/uploadaudio")
        def uploadaudio(file: UploadFile = File(...)):
            try:
                audio_content = file.read()
                transcription_result = transcribe_audio(audio_content)
                generated_messages = []
                for model_chat in model_chats:
                    generated_message = model_chat.generate_response(
                        transcription_result
                    )
                    generated_messages.append(generated_message)
                result = []
                for i, generated_message in enumerate(generated_messages):
                    language = detect(generated_message)
                    result.append({"message": generated_message, "language": language})
                    train_model(generated_message, language=language)
                return JSONResponse(content=result)
            except Exception as e:
                logger.error(f"Error in upload audio API: {e}")

        @app.get("/auto_complete")
        async def auto_complete(
            q: str = Query(..., min_length=1, max_length=518899999992, title="query")
        ):
            try:
                for dataset_name in list_datasets():
                    try:
                        datasets_dict[dataset_name] = load_dataset(dataset_name)
                        create_training_data(dataset_name, q)
                    except Exception as e:
                        logger.error(f"Error loading dataset {dataset_name}: {e}")
                with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    model_futures = [
                        executor.submit(train_model, model_name)
                        for model_name in model_names
                    ]
                    trained_models = [future.result() for future in model_futures]
                generated = model.generate(q)
                result_dict = generated[0]
                logger.debug(f"Successfully auto complete, q:{q}, res:{result_dict}")
                return result_dict
            except Exception as e:
                logger.error(f"Ignored error in auto complete: {e}")

        @app.post("/api/live_callx")
        def live_callx(text: str = Form(...)):
            try:
                filename = "response.wav"
                tts = gTTS(text=text, lang="en")
                tts.save(filename)
                return StreamingResponse(open(filename, "rb"), media_type="audio/wav")
            except Exception as e:
                logger.error(f"Error in live call: {e}")

        @app.post("/api/call")
        def call(request_data, model_chats):
            try:
                if "text" in request_data:
                    text = request_data["text"]
                    filename = "response.wav"
                    tts = gTTS(text=text, lang="en")
                    tts.save(filename)
                    return StreamingResponse(
                        open(filename, "rb"), media_type="audio/wav"
                    )
                elif "audio" in request_data:
                    audio = request_data["audio"]
                    transcription_result = transcribe_audio(audio)
                    generated_messages = []
                    for model_chat in model_chats:
                        generated_message = model_chat.generate_response(
                            transcription_result
                        )
                        generated_messages.append(generated_message)
                    sentiment_analyzer = SentimentIntensityAnalyzer()
                    result = []
                    for i, generated_message in enumerate(generated_messages):
                        sentiment_score = sentiment_analyzer.polarity_scores(
                            generated_message
                        )["compound"]
                        language = detect(generated_message)
                        result.append(
                            {
                                "message": generated_message,
                                "sentiment": sentiment_score,
                                "language": language,
                            }
                        )
                    return JSONResponse(content=result)
                elif "request" in request_data:
                    user_input = request_data["request"]
                    generated_messages = []
                    for model_chat in model_chats:
                        generated_message = model_chat.generate_response(user_input)
                        generated_messages.append(generated_message)
                    sentiment_analyzer = SentimentIntensityAnalyzer()
                    result = []
                    for i, generated_message in enumerate(generated_messages):
                        sentiment_score = sentiment_analyzer.polarity_scores(
                            generated_message
                        )["compound"]
                        language = detect(generated_message)
                        result.append(
                            {
                                "message": generated_message,
                                "sentiment": sentiment_score,
                                "language": language,
                            }
                        )
                    return JSONResponse(content=result)
                elif request == "call_text":
                    filename = "response.wav"
                    tts = gTTS(text=text, lang="en")
                    tts.save(filename)
                    return StreamingResponse(
                        open(filename, "rb"), media_type="audio/wav"
                    )
                elif request == "live_call":
                    filename = "response.wav"
                    tts = gTTS(text=text, lang="en")
                    tts.save(filename)
                    return StreamingResponse(
                        open(filename, "rb"), media_type="audio/wav"
                    )
                elif request == "live_interaction":
                    generated_messages = []
                    for model_chat in model_chats:
                        generated_message = model_chat.generate_response(text)
                        generated_messages.append(generated_message)
                    sentiment_analyzer = SentimentIntensityAnalyzer()
                    result = []
                    for generated_message in generated_messages:
                        sentiment_score = sentiment_analyzer.polarity_scores(
                            generated_message
                        )["compound"]
                        language = detect(generated_message)
                        result.append(
                            {
                                "message": generated_message,
                                "sentiment": sentiment_score,
                                "language": language,
                            }
                        )
                    return JSONResponse(content=result)
                else:
                    raise HTTPException(status_code=400, detail="Invalid request data")
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error processing request: {e}"
                )
                pass

def start_uvicon():
    app = FastAPI()
    port = int(os.environ.get("PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=port)

        

if __name__ == "__main__":
    while True:
        try:
            uvicorn_thread = threading.Thread(target=start_uvicon)
            uvicorn_thread.start()

            main()  # Esto se ejecutará en segundo plano después de iniciar uvicorn completamente
            uvicorn_thread.join()  # Espera a que uvicorn termine (esto no detiene la ejecución de main())
            
        except (AssertionError, AttributeError, BufferError, EOFError, ImportError, IndexError, KeyError, KeyboardInterrupt, MemoryError, NameError, NotImplementedError, OSError, OverflowError, ReferenceError, RuntimeError, StopIteration, SyntaxError, IndentationError, TabError, SystemError, SystemExit, TypeError, UnboundLocalError, UnicodeError, ValueError, ZeroDivisionError, ArithmeticError, FloatingPointError, ModuleNotFoundError, LookupError, EnvironmentError, IOError, FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, FileExistsError, InterruptedError, ProcessLookupError, TimeoutError, ConnectionError, BlockingIOError, BrokenPipeError, ChildProcessError, GeneratorExit, RecursionError, ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError, UnicodeEncodeError, UnicodeDecodeError, UnicodeTranslateError, Warning, UserWarning, DeprecationWarning, PendingDeprecationWarning, SyntaxWarning, RuntimeWarning, FutureWarning, ImportWarning, UnicodeWarning, BytesWarning, ResourceWarning) as e:
            traceback.print_exc()
            logger.error(f"Ignored error in main loop: {e}")
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Unhandled exception in main loop: {e}")
            raise  # Re-raise the exception for further handling or
