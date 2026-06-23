import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Define a Knowledge Base of FAQs (Questions & Answers)
FAQ_DATA = {
    "What is CodeAlpha?": "CodeAlpha is a leading software development company dedicated to driving innovation and excellence across emerging technologies.",
    "What tracks are available in the internship?": "CodeAlpha offers internships in various domains including Artificial Intelligence, Web Development, Mobile App Development, and Cyber Security.",
    "What perks do interns receive?": "Interns receive an Offer Letter, a QR-verified Completion Certificate, a Unique ID Certificate, a Letter of Recommendation based on performance, and placement support.",
    "How do I submit my completed tasks?": "You must upload your complete source code to GitHub in a specifically named repository, post a video explanation on LinkedIn, and submit the links via the official form shared in your WhatsApp group.",
    "Can I submit only one task to pass?": "No. Submitting only one task is considered incomplete, and certificates will not be issued. You must complete a minimum of two or three tasks.",
    "Who do I contact for support?": "You can reach out via email at services@codealpha.tech or contact them on WhatsApp at +91 9336576683."
}

# Extract lists of questions and answers
faq_questions = list(FAQ_DATA.keys())
faq_answers = list(FAQ_DATA.values())

def preprocess_text(text):
    """Tokenizes, cleans punctuation, and removes stopwords from text using basic Python."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation manually
    for punct in string.punctuation:
        text = text.replace(punct, " ")
        
    # Split by spaces (this replaces word_tokenize)
    tokens = text.split()
    
    # A small manual list of common English stopwords to avoid relying on nltk.corpus
    stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'do', 'i', 'how', 'my', 'can', 'who'}
    
    cleaned_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(cleaned_tokens)

def get_bot_response(user_query):
    """Matches user query with the most similar FAQ using TF-IDF and Cosine Similarity."""
    # Preprocess the user query and all FAQ questions
    cleaned_questions = [preprocess_text(q) for q in faq_questions]
    cleaned_user_query = preprocess_text(user_query)
    
    # If user input is empty after cleaning
    if not cleaned_user_query.strip():
        return "I'm sorry, I didn't quite catch that. Could you please rephrase your question?"

    # Combine questions for vectorization
    all_texts = cleaned_questions + [cleaned_user_query]

    # 2. Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # 3. Calculate Cosine Similarity between user query (last row) and FAQ questions
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    # Find the index of the highest similarity match
    best_match_idx = similarity_scores.argmax()
    highest_score = similarity_scores[best_match_idx]

    # Define a confidence threshold (e.g., 0.2) to prevent completely wrong answers
    if highest_score > 0.2:
        return faq_answers[best_match_idx]
    else:
        return "I'm sorry, I couldn't find an exact match for your question in our system. Please try asking about perks, submissions, or company details!"

if __name__ == "__main__":
    print("====================================================")
    # Simple Chat UI loop inside the terminal
    print("CodeAlpha AI FAQ Chatbot is online! (Type 'quit' to exit)")
    print("====================================================")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye! Good luck with your internship tasks!")
            break
            
        response = get_bot_response(user_input)
        print(f"Bot: {response}")