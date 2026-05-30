from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
llm = ChatOpenAI(model="gpt-4o-mini")
video_id = "0CmtDk-joT4" ## not url only id
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])

    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)
except TranscriptsDisabled:
    print("No caption available for this video")    
## create a chunk of 
splliters = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
chunks = splliters.create_documents([transcript])
print(chunks)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_id = vector_store.index_to_docstore_id
print(vector_id)

