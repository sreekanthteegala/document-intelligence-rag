from app.core.vectorstore import load_faiss
import re

_llm = None
_summarizer = None


def get_llm():
    """Text generation model for Q&A"""
    global _llm
    if _llm is None:
        from transformers import pipeline
        _llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1
        )
    return _llm


def get_summarizer():
    """Dedicated summarization model"""
    global _summarizer
    if _summarizer is None:
        from transformers import pipeline
        # Using BART - much better for summarization
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
    return _summarizer


def is_summary_question(question: str) -> bool:
    """Detect if user wants a summary"""
    q = question.lower()
    return any(
        phrase in q
        for phrase in [
            "what is this pdf about",
            "what is this document about",
            "what is this paper about",
            "what does this pdf say",
            "what does this document say",
            "summary",
            "summarize",
            "overview",
            "about this",
            "tell me about"
        ]
    )


def clean_email_noise(text: str) -> str:
    """Remove email headers, footers, and repetitive noise"""
    
    # Remove email headers
    text = re.sub(r'From:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'To:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Subject:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Date:.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', text)
    
    # Remove email footers and common phrases
    noise_patterns = [
        r'Please do not reply.*?monitored\.?',
        r'Do not reply.*?monitored\.?',
        r'To stop receiving.*?Profile\.?',
        r'unsubscribe.*?',
        r'privacy statement.*?',
        r'microsoft corporation.*?',
        r'one microsoft way.*?',
        r'user profile.*?',
        r'Dear.*?Greetings.*?\.',
        r'With regards,.*',
        r'Best regards,.*',
        r'Sincerely,.*',
        r'Organizing Committee.*',
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()


def extract_meaningful_content(text: str) -> str:
    """Extract the actual content from noisy text"""
    
    # Clean email noise first
    text = clean_email_noise(text)
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Filter out noise sentences
    meaningful = []
    skip_phrases = [
        'dear author', 'greetings from', 'please do not reply',
        'organizing committee', 'user profile', 'future references',
        'if you have any questions', 'look forward', 'with regards',
        'follow the', 'check the box', 'stop receiving'
    ]
    
    for sentence in sentences:
        s = sentence.strip()
        if len(s) < 15:  # Too short
            continue
        
        # Skip noise sentences
        if any(phrase in s.lower() for phrase in skip_phrases):
            continue
        
        # Skip sentences with only numbers/dates
        if re.match(r'^[\d\s/:.-]+$', s):
            continue
        
        meaningful.append(s)
    
    # Join meaningful sentences
    content = '. '.join(meaningful)
    
    # If content is too short, return original cleaned text
    if len(content) < 50:
        return text
    
    return content


def answer_question(question: str):
    db = load_faiss()

    # 🔹 SUMMARY MODE
    if is_summary_question(question):
        # Get more chunks for better context
        docs = db.similarity_search("main content key information", k=10)
        
        if not docs:
            return "No document content found.", []
        
        # Combine all text
        raw_text = " ".join(d.page_content for d in docs)
        
        # Extract meaningful content
        cleaned_text = extract_meaningful_content(raw_text)
        
        # Limit to reasonable length for summarization
        max_length = 1024
        if len(cleaned_text) > max_length:
            cleaned_text = cleaned_text[:max_length]
        
        # If cleaned text is too short, there's no content
        if len(cleaned_text) < 100:
            return "Unable to generate a meaningful summary from the document.", []
        
        try:
            # Use dedicated summarizer
            summarizer = get_summarizer()
            
            summary = summarizer(
                cleaned_text,
                max_length=130,
                min_length=40,
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            
            # Clean up summary
            summary = summary.strip()
            if not summary.endswith('.'):
                summary += '.'
            
            return summary, []
            
        except Exception as e:
            print(f"Summarization error: {e}")
            
            # Fallback: Use extractive summary (first few meaningful sentences)
            sentences = re.split(r'[.!?]+', cleaned_text)
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
            
            if meaningful_sentences:
                fallback_summary = '. '.join(meaningful_sentences) + '.'
                return fallback_summary, []
            
            return "Unable to generate summary. Please try a specific question about the document.", []

    # 🔹 NORMAL QA MODE
    docs = db.similarity_search(question, k=4)
    
    if not docs:
        return "I don't have enough information to answer this question.", []
    
    # Clean context
    context_parts = []
    for doc in docs:
        cleaned = extract_meaningful_content(doc.page_content)
        if cleaned and len(cleaned) > 30:
            context_parts.append(cleaned)
    
    context = " ".join(context_parts[:3])  # Limit to top 3
    
    if not context:
        return "I don't have relevant information to answer this question.", []
    
    llm = get_llm()
    
    prompt = f"""Answer the question based on the following context. Be concise and specific.

Context: {context}

Question: {question}

Answer:"""
    
    try:
        output = llm(
            prompt,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True
        )[0]["generated_text"]
        
        output = output.strip()
        
        # Check for non-answers
        if any(phrase in output.lower() for phrase in ['i don\'t know', 'not mentioned', 'no information']):
            return "I don't have enough information to answer this question.", []
        
        sources = [doc.page_content[:200] for doc in docs[:2]]
        
        return output, sources
        
    except Exception as e:
        print(f"Generation error: {e}")
        return "Error generating answer. Please try rephrasing your question.", []