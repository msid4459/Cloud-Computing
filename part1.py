import csv
import pandas as pd
from pyspark import SparkContext
import argparse
from pyspark.sql.functions import col, asc
from pyspark import SQLContext

if __name__ == "__main__":
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
    # Load data
    
    train = sqlContext.read.format('com.databricks.spark.csv').options(header='true').option("delimiter", "\t").load(input_path + "amazon_reviews_us_Music_v1_00.tsv")
    
    
    totalnoofreviews  = train.select('review_id').count()
    noofuniqueusers = train.select('customer_id').distinct().count()
    noofuniqueproducts = train.select('product_id').distinct().count()
    v4 = train.groupby('customer_id').count()
    largestnoofysingleuser = v4.agg({"count": "max"}).collect()[0][0]
    v6 = v4.sort(v4["count"].desc())
    v9 = v6.collect()
    top10usercount = v6.show(10)
    #median
    if len(v9)%2 == 0 :
        v7 =  v9[(len(v9))//2]['count']
    if len(v9)%2 == 1 :
        v7 = (v9[(len(v9))//2]['count'] + v9[(len(v9))//2-1]['count']) / 2
    # the largest number of reviews written for a single product
    Product_review_count = train.groupby('product_id').count()
    Largest_review_product_count = Product_review_count.agg({"count": "max"}).collect()[0][0]
    #the top 10 products ranked by the number of reviews they have
    sortedlistofproductreviews = Product_review_count.sort(Product_review_count['count'].desc())
    top10productreviewcount = sortedlistofproductreviews.show(10)
    #the median number of reviews a product has
    listofreview = sortedlistofproductreviews.collect()
    if len(listofreview)%2 == 0 :
        medianofproduct =  listofreview[(len(listofreview))//2]['count']
    if len(listofreview)%2 == 1 :
        medianofproduct = (listofreview[(len(listofreview))//2]['count'] + listofreview[(len(listofreview))//2-1]['count']) / 2
    
    #part 1 1 count
    print(totalnoofreviews)
    print(noofuniqueusers)
    print(noofuniqueproducts)
    # the largest number of reviews published by a single user
    print(largestnoofysingleuser)
    #the top 10 users ranked by the number of reviews they publish
    print(top10usercount)
    #median
    print(v7)

    print(Largest_review_product_count)
    print(top10productreviewcount)
    print(medianofproduct)


    