import gensim
from pyspark import (SparkContext, SQLContext)
import argparse
import numpy


# Returns a review id dictionary
# Format: dict[sentence] = id
def create_reviews(review_set):
    review_dict = {}
    for review in review_set:
        review_sentences = review['review_body'].strip().split('.')
        for sentence in review_sentences:
            # Append to review id dictionary
            review_dict[sentence] = review['review_id']
    return review_dict


# Returns a list of sentences
# Format: string
def create_sentences(review_set):
    sentences = []
    for review in review_set:
        review_sentences = review['review_body'].strip().split('.')
        for sentence in review_sentences:
            # Append to sentence list
            sentences.append(sentence)
    return sentences


# Returns a list of words, necessary for word2vec model
# Format: list(string)
def create_words(review_set):
    words = []
    for review in review_set:
        review_sentences = review['review_body'].strip().split('.')
        for sentence in review_sentences:
            # Append to word list
            word = [sentence]
            words.append(word)
    return words


# Train word2vec model and generate vectors
def create_vectors(word_set):
    model = gensim.models.Word2Vec(word_set, min_count=1, size=100, window=5)
    vectors = model.wv
    del model
    return vectors


# Generate distance dictionary
def generate_distances(sentence_set, vector_set):
    dist_dict = {}
    for point_A in sentence_set:
        dist_dict[point_A] = []
        # Calculate cosine distance for each sentence
        # Format: dict[point A] = [(point B, cos_dist), (point C, cost_dist), ....]
        for point_B in sentence_set:
            # Store vectors of two points
            vA = vector_set.get_vector(point_A)
            vB = vector_set.get_vector(point_B)
            # Calculate cosine distance
            cos_sim = numpy.dot(vA, vB) / (numpy.sqrt(numpy.dot(vA, vA)) * numpy.sqrt(numpy.dot(vB, vB)))
            cos_dist = 1 - cos_sim
            # Append (point B, cosine distance) tuple to dict[point A]
            dist_dict[point_A].append((point_B, cos_dist))
    return dist_dict


# Calculate distance average
def calc_avg(sentence_set, distance_set):
    avg_dict = {}
    for point_A in sentence_set:
        # Calculate cosine distance average for each point
        sum = 0
        for point_B, cos_dist in distance_set[point_A]:
            sum += cos_dist
        avg = sum / len(distance_set[point_A])
        avg_dict[point_A] = avg
    return avg_dict


# Returns the class center
# Format: tuple(sentence, avg)
def find_center(sentence_set, average_set):
    center_sent = ""
    center_avg = 2
    for sentence in sentence_set:
        if average_set[sentence] < center_avg:
            center_sent = sentence
            center_avg = average_set[sentence]
    return center_sent, center_avg


# Returns the closest neighbours
# Format: list(sentence)
def find_neighbours(average_set, center_avg):
    neighbours = []
    avg_list = list(average_set.values())
    for index in range(10):
        next_neighbour = min(avg_list, key=lambda x: abs(x - center_avg))
        avg_list.remove(next_neighbour)
        for sentence, avg in average_set.items():
            if avg == next_neighbour:
                neighbours.append(sentence)
                break
    return neighbours


# Returns reviews
# Format: list(tuple(id, sentence))
def find_reviews(review_set, center_sent, neighbour_set):
    review_list = list()
    review_list.append((review_set[center_sent], center_sent))
    for neighbour in neighbour_set:
        review_list.append((review_set[neighbour], neighbour))
    return review_list


# Print reviews
def print_reviews(reviews, title):
    print(title + "\n")
    for review_id, review_sent in reviews:
        print(str(review_id) + " : " + str(review_sent) + "\n")


# Main
if __name__ == "__main__":
    # Set up argument parser
    sc = SparkContext(appName="Assignment 2")
    sqlContext = SQLContext(sc)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the input path",
                        default='data/')
    parser.add_argument("--output", help="the output path", 
                        default='output')
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    # Train dataset
    dataset = sqlContext.read.format('com.databricks.spark.csv').options(header='true').option("delimiter", "\t").load(input_path + "amazon_reviews_us_Music_v1_00.tsv")

    # Count number of product reviews
    product_review_count = dataset.groupby('product_id').count()

    # Top 10 products ranked by number of reviews
    top_10_product_review_count = product_review_count.sort(product_review_count['count'].desc()).take(10)

    # Select review classes for the first product on the list
    pos_reviews = dataset.where(dataset.star_rating >= 4).where(dataset.product_id == top_10_product_review_count[0]['product_id']).collect()
    neg_reviews = dataset.where(dataset.star_rating <= 2).where(dataset.product_id == top_10_product_review_count[0]['product_id']).collect()

    # Create a dict for review body(key)and review id(value)
    pos_review_dict = create_reviews(pos_reviews)
    neg_review_dict = create_reviews(neg_reviews)

    # Create a list of 'words' for word2vec model
    pos_sentences = create_sentences(pos_reviews)
    neg_sentences = create_sentences(neg_reviews)

    # Create a list of review sentences
    pos_words = create_words(pos_reviews)
    neg_words = create_words(neg_reviews)

    # Create a list of vectors
    pos_vectors = create_vectors(pos_words)
    neg_vectors = create_vectors(neg_words)

    # Store cosine distances
    pos_dist_dict = generate_distances(pos_sentences, pos_vectors)
    neg_dist_dict = generate_distances(neg_sentences, neg_vectors)

    # Store cosine distance averages
    pos_avg_dict = calc_avg(pos_sentences, pos_dist_dict)
    neg_avg_dict = calc_avg(neg_sentences, neg_dist_dict)

    # Store the center sentence and average
    pos_center_sent, pos_center_avg = find_center(pos_sentences, pos_avg_dict)
    neg_center_sent, neg_center_avg = find_center(neg_sentences, neg_avg_dict)

    # Find ten closest neighbours from the center
    pos_neighbours = find_neighbours(pos_avg_dict, pos_center_avg)
    neg_neighbours = find_neighbours(neg_avg_dict, neg_center_avg)

    # Store reviews
    pos_review_list = find_reviews(pos_review_dict, pos_center_sent, pos_neighbours)
    neg_review_list = find_reviews(neg_review_dict, neg_center_sent, neg_neighbours)

    # Print reviews
    print_reviews(pos_review_list, "Positive Reviews")
    print_reviews(neg_review_list, "Negative Reviews")
