import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import gensim
from gensim.summarization import summarize
    # Instantiating the model and tokenizer with gpt-2
    
def summarizer(rawdocs):
 
    tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
    model=GPT2LMHeadModel.from_pretrained('gpt2')
    # Encoding text to get input ids & pass them to model.generate()
    inputs=tokenizer.batch_encode_plus([rawdocs],return_tensors='pt',max_length=512,truncation=True)
    summary_ids=model.generate(inputs['input_ids'],early_stopping=True)
    GPT_summary=tokenizer.decode(summary_ids[0],skip_special_tokens=True)
    #print(GPT_summary)
    # Passing the text corpus to summarizer 
    short_summary = summarize(rawdocs)
    #print(short_summary)
    # Summarization by ratio
    summary_by_ratio=summarize(rawdocs,ratio=0.5)
    print(summary_by_ratio)
    # Summarization by word count
    summary_by_word_count=summarize(rawdocs,word_count=60)
    #print(summary_by_word_count)
    return summary_by_word_count,GPT_summary
