import pandas as pd
from sentence_transformers import SentenceTransformer


# Define the clean and noisy text samples
noisy_samples = [
   """<p>Welcome to our site!!! Visit <a href="http://example.com">this link</a> for more information. Data quality? It's crucial: removing noise like tags, symbols, and irrelevant links improves text accuracy. Our latest research at <b>Example University</b> shows how noise impacts retrieval models. 100% proven methods!</p>""",
   """Contact us at john.doe@example.com or call 123-456-7890 for support. Data preprocessing includes removing personally identifiable information (PII) to ensure privacy. This is essential in sectors like healthcare and finance. Studies show that removing PII enhances text processing models' efficiency. Did you know? 50% of data breaches are caused by exposed PII!""",
   """Text CLEANING!!! is CRITICAL for IMPROVING machine learning MODEL accuracy!!! In NLP tasks, handling excessive punctuation, mixed casing, and symbols like @, #, & is key. Proper data preparation can significantly affect model outcomes... Let's dive deep into HOW TO achieve the BEST RESULTS.""",
   """Machine learning is transforming industries. Machine learning is transforming industries. In the age of AI, removing repetitive content is crucial for efficient processing. Repetition can distort results and lead to poor model performance. Machine learning is transforming industries. Removing duplicated data is essential for accuracy.""",
   """It's essential to preprocess text data. Inconsistent encoding and mixed language inputs, such as "Este es un ejemplo de datos mezclados," can cause issues. Preprocessing helps standardize data, ensuring models perform optimally. Text processing also includes fixing broken unicode characters like â€“ and handling language-specific nuances."""
]


clean_samples = [
   """Welcome to our site. Data quality is crucial: removing noise like tags, symbols, and irrelevant links improves text accuracy. Our latest research at Example University shows how noise impacts retrieval models with proven methods.""",
   """Data preprocessing includes removing personally identifiable information (PII) to ensure privacy. This is essential in sectors like healthcare and finance. Studies show that removing PII enhances text processing models' efficiency.""",
   """Text cleaning is critical for improving machine learning model accuracy. In NLP tasks, handling punctuation, mixed casing, and symbols is key. Proper data preparation can significantly affect model outcomes.""",
   """Machine learning is transforming industries. In the age of AI, removing repetitive content is crucial for efficient processing. Repetition can distort results and lead to poor model performance. Removing duplicated data is essential for accuracy.""",
   """It's essential to preprocess text data. Inconsistent encoding and mixed language inputs can cause issues. Preprocessing helps standardize data, ensuring models perform optimally. Text processing also includes fixing broken unicode characters and handling language-specific nuances."""
]


# Define queries relevant to the samples
queries = [
   "How noise impacts retrieval models",
   "Removing PII to enhance text processing",
   "Improving machine learning model accuracy",
   "Removing repetitive content in machine learning",
   "Standardizing data for text preprocessing"
]


# Load a pre-trained model from Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')


# Generate embeddings for noisy, clean samples, and queries
noisy_embeddings = model.encode(noisy_samples)
clean_embeddings = model.encode(clean_samples)
query_embeddings = model.encode(queries)


# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
   dot_product = sum(a * b for a, b in zip(vec1, vec2))
   magnitude1 = sum(a**2 for a in vec1) ** 0.5
   magnitude2 = sum(b**2 for b in vec2) ** 0.5
   if magnitude1 == 0 or magnitude2 == 0:
       return 0
   return dot_product / (magnitude1 * magnitude2)


# Calculate similarities for both noisy and clean data
noisy_similarities = [[cosine_similarity(query, sample) for sample in noisy_embeddings] for query in query_embeddings]
clean_similarities = [[cosine_similarity(query, sample) for sample in clean_embeddings] for query in query_embeddings]


# Prepare data for CSV export
rows = []


for i, query in enumerate(queries):
   for j in range(len(noisy_similarities[i])):
       rows.append({
           "Query": query,
           "Sample": f"Sample {j+1}",
           "Noisy Similarity": noisy_similarities[i][j],
           "Clean Similarity": clean_similarities[i][j],
           "Difference": clean_similarities[i][j] - noisy_similarities[i][j],
           "Matched Text (Noisy)": noisy_samples[j],
           "Matched Text (Clean)": clean_samples[j]
       })


# Convert to DataFrame
df = pd.DataFrame(rows)


# Save to CSV
csv_file_path = "similarity_analysis_with_text.csv"
df.to_csv(csv_file_path, index=False)


print(f"CSV file saved to {csv_file_path}")
