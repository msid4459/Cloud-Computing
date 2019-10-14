#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv(r'C:\Users\msid4\OneDrive\Desktop\cc\music.tsv',sep='\t',encoding='utf-8',error_bad_lines=False)


# In[3]:


df.head()


# In[90]:


df.shape


# In[4]:


import findspark


# In[5]:


findspark.init('C:\spark')


# In[6]:


from pyspark.sql import SparkSession


# In[7]:


from pyspark import SparkConf,SparkContext


# In[8]:


from pyspark.sql import SQLContext


# In[9]:


conf=SparkConf().setMaster("local").setAppName("Assignment2")
sc=SparkContext(conf=conf)


# In[10]:


sqlContext = SQLContext(sc)


# In[11]:


data1 = sqlContext.read.format("com.databricks.spark.csv").options(header='true').option("delimiter", "\t").load(r"C:\Users\msid4\OneDrive\Desktop\cc\music.tsv")


# In[12]:


v2=data1.select('review_body','customer_id')


# In[13]:


v2.show()


# In[176]:


v3 = v2.groupBy("customer_id").count()


# In[177]:


v3.printSchema


# In[ ]:


v3.registerTempTable("v3")


# In[ ]:


sqlContext.sql("SELECT percentile_approx(count, 0.5) as median FROM v3").show()


# In[15]:


v4=data1.select('review_body','product_id')
v4.registerTempTable("v4")


# In[16]:


v6=v4.select('review_body','product_id').groupBy("product_id").count()
v6.show()


# In[17]:


v6.registerTempTable("v6")


# In[124]:


median_product=sqlContext.sql("SELECT percentile_approx(count, 0.5) as median FROM v6")


# In[107]:


from pyspark.sql import functions as F 


# In[22]:


v6=v6.withColumnRenamed("product_id", "productid")


# In[23]:


data1 = data1.join(v6, data1.product_id == v6.productid)
data1.show()


# In[29]:


data1_output=data1.where('count>=2')


# In[30]:


v10=data1_output.select('product_id','review_body','count')
v10.registerTempTable("v10")


# In[33]:


top_10_products=sqlContext.sql("SELECT review_body,product_id FROM v10  order by count desc limit 10 ").show()


# In[ ]:





# In[78]:


regex_Tokenizer=RegexTokenizer(inputCol='review_body', outputCol='sentences',pattern='\\.')


# In[79]:


count_tokens=udf(lambda sentences:len(sentences),IntegerType())


# In[81]:


reg_tokenized=regex_Tokenizer.transform(data1_output)


# In[ ]:


reg_tokenized.show()


# In[ ]:


data2=reg_tokenized.withColumn('tokens',count_tokens(col('sentences'))).show()


# In[ ]:


filtered_data=data2.where('tokens>=2')

