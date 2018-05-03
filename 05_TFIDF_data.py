#  TF-IDF
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

from pyspark.ml.feature import HashingTF, IDF
sentences_df
# DataFrame[id: bigint, sentence: string]
sentences_df.take(1)
# [Row(id=1, sentence=u'This is an introduction to Spark MLlib')]
sent_tokenized_df.take(1)
# [Row(id=1, sentence=u'This is an introduction to Spark MLlib', words=[u'this', u'is', u'an', u'introduction', u'to', u'spark', u'mllib'])]

hashingTF = HashingTF(inputCol="words",outputCol="rawFeatures",numFeatures=20)
sent_hfTF_df = hashingTF.transform(sent_tokenized_df)
sent_hfTF_df.show(10,False)
# +---+----------------------------------------------------------+------------------------------------------------------------------+------------------------------------------------------+
# |id |sentence                                                  |words                                                             |rawFeatures                                           |
# +---+----------------------------------------------------------+------------------------------------------------------------------+------------------------------------------------------+
# |1  |This is an introduction to Spark MLlib                    |[this, is, an, introduction, to, spark, mllib]                    |(20,[1,5,6,8,12,13],[2.0,1.0,1.0,1.0,1.0,1.0])        |
# |2  |MLlib includes libraries for classification and regression|[mllib, includes, libraries, for, classification, and, regression]|(20,[1,6,9,12,13,15,16],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])|
# |3  |It also contains supporting tools for pipelines           |[it, also, contains, supporting, tools, for, pipelines]           |(20,[0,8,10,12,15,16],[1.0,1.0,1.0,1.0,1.0,2.0])      |
# +---+----------------------------------------------------------+------------------------------------------------------------------+------------------------------------------------------+

sent_hfTF_df.take(1)
# [Row(id=1, sentence=u'This is an introduction to Spark MLlib', words=[u'this', u'is', u'an', u'introduction', u'to', u'spark', u'mllib'], rawFeatures=SparseVector(20, {1: 2.0, 5: 1.0, 6: 1.0, 8: 1.0, 12: 1.0, 13: 1.0}))]
idf = IDF(inputCol='rawFeatures',outputCol='idf_features')
idfModel = idf.fit(sent_hfTF_df)
tfidf_df = idfModel.transform(sent_hfTF_df)
tfidf_df.show(10,False)
# +---+----------------------------------------------------------+------------------------------------------------------------------+------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
# |id |sentence                                                  |words                                                             |rawFeatures                                           |idf_features                                                                                                                                         |
# +---+----------------------------------------------------------+------------------------------------------------------------------+------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
# |1  |This is an introduction to Spark MLlib                    |[this, is, an, introduction, to, spark, mllib]                    |(20,[1,5,6,8,12,13],[2.0,1.0,1.0,1.0,1.0,1.0])        |(20,[1,5,6,8,12,13],[0.5753641449035617,0.6931471805599453,0.28768207245178085,0.28768207245178085,0.0,0.28768207245178085])                         |
# |2  |MLlib includes libraries for classification and regression|[mllib, includes, libraries, for, classification, and, regression]|(20,[1,6,9,12,13,15,16],[1.0,1.0,1.0,1.0,1.0,1.0,1.0])|(20,[1,6,9,12,13,15,16],[0.28768207245178085,0.28768207245178085,0.6931471805599453,0.0,0.28768207245178085,0.28768207245178085,0.28768207245178085])|
# |3  |It also contains supporting tools for pipelines           |[it, also, contains, supporting, tools, for, pipelines]           |(20,[0,8,10,12,15,16],[1.0,1.0,1.0,1.0,1.0,2.0])      |(20,[0,8,10,12,15,16],[0.6931471805599453,0.28768207245178085,0.6931471805599453,0.0,0.28768207245178085,0.5753641449035617])                        |
# +---+----------------------------------------------------------+------------------------------------------------------------------+------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------+
tfidf_df.take(1)
# [Row(id=1, sentence=u'This is an introduction to Spark MLlib', words=[u'this', u'is', u'an', u'introduction', u'to', u'spark', u'mllib'], rawFeatures=SparseVector(20, {1: 2.0, 5: 1.0, 6: 1.0, 8: 1.0, 12: 1.0, 13: 1.0}), idf_features=SparseVector(20, {1: 0.5754, 5: 0.6931, 6: 0.2877, 8: 0.2877, 12: 0.0, 13: 0.2877}))]
