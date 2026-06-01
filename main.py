from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv(override=True)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

## how to see chunks
chunk_vector = vector_store.get_by_ids(['aa512892-df19-43db-ad9c-535ae233ae1d'])
print(chunk_vector)
## generate a retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
result = retriever.invoke('what is book')
print(result)

## Step 3 - Augmentation

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})

answer = llm.invoke(final_prompt)
print(answer.content)


## bulid by a chain 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
result = main_chain.invoke("can you summarize the video")
print(result)