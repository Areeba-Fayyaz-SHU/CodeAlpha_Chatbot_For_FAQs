# CodeAlpha - Chatbot for FAQs

An intelligent, NLP-driven FAQ Chatbot built as part of the CodeAlpha Artificial Intelligence Internship. This project processes user queries dynamically and matches them against a predefined knowledge base of frequently asked questions to return accurate, contextual answers instantly.

## Features
- **Dynamic Text Preprocessing:** Custom lightweight NLP pipeline that tokenizes text, normalizes cases, and strips out structural punctuation or common English stopwords.
- **TF-IDF Vectorization:** Transforms raw text queries into statistical frequency vectors, mapping the importance of key phrases mathematically.
- **Cosine Similarity Matching:** Computes the angular distance between the user's vector and the FAQ dataset vectors to find the closest logical match.
- **Fallback Guardrails:** Implements a confidence threshold to elegantly handle completely unrelated text inputs or casual greetings.

## How the NLP Text Vectorization Works

To match a user's question with the correct answer without hardcoding static keywords, the system uses an advanced Information Retrieval (IR) pipeline:

1. **Text Preprocessing:** Raw strings are converted to lowercase, stripped of all punctuation characters, and split into individual word tokens.
2. **TF-IDF (Term Frequency-Inverse Document Frequency):** The tokens are converted into numerical feature vectors. Term Frequency tracks how frequently a word appears in a query, while Inverse Document Frequency down-weights generic words across the text database, highlighting unique intent tokens.
3. **Cosine Similarity:** The script calculates the cosine of the angle between the newly created user query vector ($A$) and each pre-existing FAQ question vector ($B$). The formula used determines structural similarity:
   $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
4. **Result Selection:** The system extracts the FAQ answer linked to the question that scored the highest similarity coefficient.

## Technologies Used
- Python 3
- Scikit-Learn (for TF-IDF Vectorization & Cosine Similarity)
- Built-in Python String Processing Mechanics

## How to Run Locally

1. Clone this repository or download the files into a single folder.
2. Install the required mathematical and vectorization packages:
```bash
   pip install scikit-learn
1. Run the interactive terminal chatbot:
      python faq_chatbot.py
2. Ask any question related to the company, rewards, submission policies, or contact support! (Type quit to exit).
