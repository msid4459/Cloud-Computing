import csv
import tensorflow as tf
from pyspark import SparkContext
import argparse
import numpy
from pyspark.sql.functions import col, asc
from pyspark import SQLContext
import tensorflow as tf
import tensorflow_hub as hub
def create_reviews_data(review_set):
    class_review_body_set = []
    review_dict = {}
    i=0
    for review in review_set:
        review_dict[i] = review['review_id']
        class_review_body_set.append(review['review_body'].strip())
        i = i+1
    return class_review_body_set,review_dict
# Returns a review id dictionary
# Format: dict[sentence] = id
def create_reviews(review_set):
    review_dict = {}
    for review in review_set:
        review_sentences = review['review_body'].strip()
        review_dict[review_sentences] = review['review_id']
    return review_dict
# Train Google's Universal encoder model and generate vectors
def create_vectors(word_set):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        vectors = session.run(embed(word_set))
    return vectors
def generate_distances(sentence_set, vector_set):
    dist_dict = {}
    for point_A in range(len(sentence_set)):
        dist_dict[point_A] = []
        # Calculate cosine distance for each sentence
        # Format: dict[point A] = [(point B, cos_dist), (point C, cost_dist), ....]
        for point_B in range(len(sentence_set)):
            # Store vectors of two points
            vA = vector_set[point_A]
            vB = vector_set[point_B]
            # Calculate cosine distance
            cos_sim = numpy.dot(vA, vB) / (numpy.sqrt(numpy.dot(vA, vA)) * numpy.sqrt(numpy.dot(vB, vB)))
            cos_dist = 1 - cos_sim
            # Append (point B, cosine distance) tuple to dict[point A]
            dist_dict[point_A].append((point_B, cos_dist))
    return dist_dict
def calc_avg(sentence_set, distance_set):
    avg_dict = {}
    for point_A in range(len(sentence_set)):
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
    for sentence in range(len(sentence_set)):
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
def print_reviews(reviews, title,sentense_list):
    print(title + "\n")
    for review_id, review_sent in reviews:
        print(str(review_id) + " : " + str(sentense_list[review_sent]) + "\n")
#Main
if __name__ == "__main__":
    # Set up argument parser
    sc = SparkContext.getOrCreate()
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
    # Create a list of review bodies
    positive_review_body_set,pos_review_dict = create_reviews_data(pos_reviews)
    negative_review_body_set,neg_review_dict = create_reviews_data(neg_reviews)

    # Create a dict for review body(key)and review id(value)
    #pos_review_dict = create_reviews(pos_reviews)
    #neg_review_dict = create_reviews(neg_reviews)

    # Create a list of vectors
    pos_vectors = create_vectors(positive_review_body_set)
    neg_vectors = create_vectors(negative_review_body_set)

    # Store cosine distances
    pos_dist_dict = generate_distances(positive_review_body_set, pos_vectors)
    neg_dist_dict = generate_distances(negative_review_body_set, neg_vectors)

    # Store cosine distance averages
    pos_avg_dict = calc_avg(positive_review_body_set, pos_dist_dict)
    neg_avg_dict = calc_avg(negative_review_body_set, neg_dist_dict)

    # Store the center sentence and average
    pos_center_sent, pos_center_avg = find_center(positive_review_body_set, pos_avg_dict)
    neg_center_sent, neg_center_avg = find_center(negative_review_body_set, neg_avg_dict)

    # Find ten closest neighbours from the center
    pos_neighbours = find_neighbours(pos_avg_dict, pos_center_avg)
    neg_neighbours = find_neighbours(neg_avg_dict, neg_center_avg)

    # Store reviews
    pos_reviews = find_reviews(pos_review_dict, pos_center_sent, pos_neighbours)
    neg_reviews = find_reviews(neg_review_dict, neg_center_sent, neg_neighbours)

    # Print reviews
    print_reviews(pos_reviews, "Positive Reviews",positive_review_body_set)
    print_reviews(neg_reviews, "Negative Reviews",negative_review_body_set)
