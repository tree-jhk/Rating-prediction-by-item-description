import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os
import random
import numpy as np


change_ml_title = [
"Alien 3",
"blood & wine",
"american dream",
"dumb & dumber",
"kama sutra: a tale of love",
"kicking and screaming",
"oscar & lucinda",
"richie rich",
"seven",
"willy wonka and the chocolate factory",
"up close and personal"
]

change_to = [
"Alien³",
"blood and wine",
"american dreamz",
"dumb and dumbe",
"kama sutra - a tale of love",
"kicking & screaming",
"oscar and lucinda",
"ri¢hie ri¢h",
"se7en",
"up close & personal",
"willy wonka & the chocolate factory"
]
change_title_dic = {k:v for k, v in zip(change_ml_title, change_to)}

def merge(
    ml_movie:pd.DataFrame=None, 
    ml_rating:pd.DataFrame=None, 
    ml_user:pd.DataFrame=None,
    tmdb_movie:pd.DataFrame=None,
    ):
    """
    기본적인 전처리를 포함해서,
    ml_100k 데이터셋과 tmdb 데이터셋 merge:
        - 같은 title & release year 기준으로 merge
    매칭된 데이터와 ml_rating 데이터셋 merge:
        - ml_100k의 movie_id 기준으로 merge
    """
    if ml_movie == None:
        ml_movie = pd.read_csv('ml-100k/movie.csv')
    if ml_rating == None:
        ml_rating = pd.read_csv('ml-100k/ratings.csv')
    if ml_user == None:
        ml_user = pd.read_csv('ml-100k/user.csv')
    if tmdb_movie == None:
        tmdb_movie = pd.read_csv('tmdb_5000_movies.csv')
    
    ml_movie.movie_title = ml_movie.movie_title.apply(lambda x: x.lower())
    tmdb_movie.title = tmdb_movie.title.apply(lambda x: x.lower())
    tmdb_movie.original_title = tmdb_movie.original_title.apply(lambda x: x.lower())
    
    tmdb_movie.loc[tmdb_movie['title'] =='america is still the place', 'original_title'] = 'america is still the place'
    tmdb_movie.loc[tmdb_movie['release_date'].isna(), 'release_date'] = '2015'

    def format_movie_title(data):
        title, date = data['title'], data['release_date']
        
        cleaned_title = re.sub(r'\([^()]*\)', '', title).strip()
        cleaned_date = re.sub(r'[^0-9-]+', '', date).strip()
        
        year = re.findall(r'\d{4}', cleaned_date)
        if year:
            year = year[0]
        else:
            year = ''
        
        formatted_title = cleaned_title + ' (' + year + ')'
        return formatted_title

    def remove_parenthesis(text):
        cleaned_text = re.sub(r'\([^()]*\)', '', text)
        return cleaned_text.strip()

    ml_movie.movie_title = ml_movie.movie_title.map(lambda x: remove_parenthesis(x))
    
    tmdb_movie.drop(columns='title', inplace=True)
    tmdb_movie.rename(columns={'original_title':'title'}, inplace=True)
    ml_movie.rename(columns={'movie_title':'title'}, inplace=True)
    
    def c(x):
        try:
            return change_title_dic[x]
        except:
            return x

    # title 추가 변경 전처리
    ml_movie.title = ml_movie.title.apply(lambda x:c(x))
    
    # tmdb에서 title 같은데, 출시일자가 달라서 다르고 줄거리는 같은 영화는 중복 제거
    tmdb_movie.drop_duplicates(['title'], keep='first', inplace=True)
    
    # ml_movie와 tmdb_movie의 title 같은 경우에 대해 rating과 함께 dataframe 합치기 
    m = pd.merge(ml_movie, tmdb_movie, how='inner', on='title')
    
    mm = pd.merge(ml_rating, m, how='inner', on='movie_id')
    
    target = pd.merge(mm, ml_user, how='inner', on='user_id')
    
    return target

def load_language_model(args, num_parameters:int=7, baseline:int=1):
    """
    baseline:
        BERT
    best:
        sentence-transformers/all-MiniLM-L6-v2
    second:
        sentence-transformers/paraphrase-MiniLM-L6-v2
    """
    if baseline:
        model = AutoModel.from_pretrained(f"bert-base-uncased")
    else:
        available_LlamaModel_parameters = [7, 13, 33, 65]
        assert num_parameters in available_LlamaModel_parameters, f"{num_parameters}B size model not exists"
        # model = AutoModel.from_pretrained(f"huggyllama/llama-{num_parameters}b") # 동작이 안 됨
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # 제일 잘 됨
        # model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2') # 세번째로 잘 됨
        # model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1') # 동작이 안 됨
        # model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2') # 두번째로 잘 됨
        # model = AutoModel.from_pretrained('LLukas22/all-mpnet-base-v2-embedding-all') # 동작이 안 됨
        # model = AutoModel.from_pretrained('EleutherAI/gpt-neo-125m') # 동작이 안 됨
        
    
    return model.to(args.device)

def load_tokenizer(args, num_parameters:int=7, baseline:int=1):
    """
    baseline:
        BERT
    best:
        sentence-transformers/all-MiniLM-L6-v2
    second:
        sentence-transformers/paraphrase-MiniLM-L6-v2
    """
    if baseline:
        tokenizer = AutoTokenizer.from_pretrained(f"bert-base-uncased")
    else:
        available_LlamaModel_parameters = [7, 13, 33, 65]
        assert num_parameters in available_LlamaModel_parameters, f"{num_parameters}B size model not exists"
        # tokenizer = AutoTokenizer.from_pretrained(f"huggyllama/llama-{num_parameters}b") # 동작이 안 됨
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2') # 제일 잘 됨
        # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2') # 세번째로 잘 됨
        # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1') # 동작이 안 됨
        # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2') # 두번째로 잘 됨
        # tokenizer = AutoTokenizer.from_pretrained('LLukas22/all-mpnet-base-v2-embedding-all') # 동작이 안 됨
        # tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m') # 동작이 안 됨
    
    return tokenizer

def text_emebdding(args, input_text:str, tokenizer, model):
    """
    input_text: string of the text
    """
    encoded_input = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt').to(args.device)

    with torch.no_grad():
        model_output = model(encoded_input)
        embedding = model_output
    
    return torch.stack(embedding, dim=0)

def setSeeds(seed=42):

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def tokenize(args, text:str, tokenizer, max_length:int=120):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    encoded_input = tokenizer.encode(
                text,
                add_special_tokens=True,
                return_tensors='pt', # 설정하면 (120) shape의 텐서로 저장함
                padding="max_length",
                max_length=max_length,
                truncation=True,
                )
    return encoded_input

class TQDMBytesReader(object):

    def __init__(self, fd, **kwargs):
        self.fd = fd
        from tqdm import tqdm
        self.tqdm = tqdm(**kwargs)

    def read(self, size=-1):
        bytes = self.fd.read(size)
        self.tqdm.update(len(bytes))
        return bytes

    def readline(self):
        bytes = self.fd.readline()
        self.tqdm.update(len(bytes))
        return bytes

    def __enter__(self):
        self.tqdm.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self.tqdm.__exit__(*args, **kwargs)