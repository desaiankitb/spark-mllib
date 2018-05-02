# tokenizer
from pyspark.ml.feature import Tokenizer
sentences_df = spark.createDataFrame([
     (1, "This is an introduction to Spark MLlib"),
     (2, "MLlib includes libraries for classification and regression"),
     (3, "It also contains supporting tools for pipelines")],
     ["id","sentence"])

sentences_df.show(10,False)
# +---+----------------------------------------------------------+
# |id |sentence                                                  |
# +---+----------------------------------------------------------+
# |1  |This is an introduction to Spark MLlib                    |
# |2  |MLlib includes libraries for classification and regression|
# |3  |It also contains supporting tools for pipelines           |
# +---+----------------------------------------------------------+

sent_token = Tokenizer(inputCol="sentence", outputCol="words")
sent_tokenized_df = sent_token.transform(sentences_df)
sent_tokenized_df.show(10,False)
# +---+----------------------------------------------------------+------------------------------------------------------------------+
# |id |sentence                                                  |words                                                             |
# +---+----------------------------------------------------------+------------------------------------------------------------------+
# |1  |This is an introduction to Spark MLlib                    |[this, is, an, introduction, to, spark, mllib]                    |
# |2  |MLlib includes libraries for classification and regression|[mllib, includes, libraries, for, classification, and, regression]|
# |3  |It also contains supporting tools for pipelines           |[it, also, contains, supporting, tools, for, pipelines]           |
# +---+----------------------------------------------------------+------------------------------------------------------------------+
