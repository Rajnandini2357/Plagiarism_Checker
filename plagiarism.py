import nltk

# Download 'punkt' tokenizer
nltk.download('punkt')

# Rest of the code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def detect_plagiarism(text1, text2):
    def preprocess_text(text):
        # Tokenization
        tokens = nltk.word_tokenize(text.lower())
        # Removing stopwords and non-alphanumeric tokens
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return ' '.join(tokens)

    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    def calculate_similarity(text1, text2):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        return similarity_matrix[0][1]

    similarity_score = calculate_similarity(preprocessed_text1, preprocessed_text2)

    # Define threshold for similarity score
    threshold = 0.7

    if similarity_score >= threshold:
        return True, similarity_score
    else:
        return False, similarity_score

if __name__ == "__main__":
    original_text = input("Enter the original text: ")
    submitted_text = input("Enter the submitted text: ")

    is_plagiarized, similarity_score = detect_plagiarism(original_text, submitted_text)

    if is_plagiarized:
        print("Plagiarism Detected!")
    else:
        print("No Plagiarism Detected.")

    print("Similarity Score:", similarity_score)
