# Normalizing data
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
df_features = spark.createDataFrame([ 
   (1, Vectors.dense([10.0, 10000.0, 1.0]),),
   (2, Vectors.dense([20.0, 30000.0, 2.0]),),
   (3, Vectors.dense([30.0, 40000.0, 3.0]),)
   ],["id","features"])
df_features
DataFrame[id: bigint, features: vector]
df_features.printSchema()
# root
#  |-- id: long (nullable = true)
#  |-- features: vector (nullable = true)
df_features.count()
# 3                                                         
df_features.show()
# +---+------------------+
# | id|          features|
# +---+------------------+
# |  1|[10.0,10000.0,1.0]|
# |  2|[20.0,30000.0,2.0]|
# |  3|[30.0,40000.0,3.0]|
# +---+------------------+
df_features.take(1)
# [Row(id=1, features=DenseVector([10.0, 10000.0, 1.0]))]
df_features.take(2)
# [Row(id=1, features=DenseVector([10.0, 10000.0, 1.0])), Row(id=2, features=DenseVector([20.0, 30000.0, 2.0]))]
featureScaler = MinMaxScaler(inputCol="features",outputCol="sfeatures") 
smodel = featureScaler.fit(df_features)
dfSfeatures.show(10,False)
# +---+------------------+----------------------------+
# |id |features          |sfeatures                   |
# +---+------------------+----------------------------+
# |1  |[10.0,10000.0,1.0]|[0.0,0.0,0.0]               |
# |2  |[20.0,30000.0,2.0]|[0.5,0.6666666666666666,0.5]|
# |3  |[30.0,40000.0,3.0]|[1.0,1.0,1.0]               |
# +---+------------------+----------------------------+
