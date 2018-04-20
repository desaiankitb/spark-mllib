>>> from pyspark.ml.linalg import Vectors
>>> from pyspark.ml.feature import VectorAssembler
>>> from pyspark.ml.clustering import KMeans
>>> cluster_df = spark.read.csv("file:///home/ankit/Documents/personal/code/linkedin-learning-downloader/out/Spark\ for\ Machine\ Learning\ &\ AI/Ex_Files_Spark_ML_AI/Exercise\ Files/Ch03/03_02", header=True, inferSchema=True)
>>> cluster_df
# DataFrame[col1: int, col2: int, col3: int]
>>> cluster_df.show(10,False)
# +----+----+----+
# |col1|col2|col3|
# +----+----+----+
# |7   |4   |1   |
# |7   |7   |9   |
# |7   |9   |6   |
# |1   |6   |5   |
# |6   |7   |7   |
# |7   |9   |4   |
# |7   |10  |6   |
# |7   |8   |2   |
# |8   |3   |8   |
# |4   |10  |5   |
# +----+----+----+
# only showing top 10 rows

>>> vectorAssembler = VectorAssembler(inputCols=["col1","col2","col3"], outputCol="features")
>>> vcluster_df = vectorAssembler.transform(cluster_df)
>>> vcluster_df
# DataFrame[col1: int, col2: int, col3: int, features: vector]
>>> vcluster_df.show(10,False)
# +----+----+----+--------------+
# |col1|col2|col3|features      |
# +----+----+----+--------------+
# |7   |4   |1   |[7.0,4.0,1.0] |
# |7   |7   |9   |[7.0,7.0,9.0] |
# |7   |9   |6   |[7.0,9.0,6.0] |
# |1   |6   |5   |[1.0,6.0,5.0] |
# |6   |7   |7   |[6.0,7.0,7.0] |
# |7   |9   |4   |[7.0,9.0,4.0] |
# |7   |10  |6   |[7.0,10.0,6.0]|
# |7   |8   |2   |[7.0,8.0,2.0] |
# |8   |3   |8   |[8.0,3.0,8.0] |
# |4   |10  |5   |[4.0,10.0,5.0]|
# +----+----+----+--------------+
# only showing top 10 rows

>>> kmeans = KMeans().setK(3)
>>> kmeans = kmeans.setSeed(1)
>>> kmodel = kmeans.fit(vcluster_df)
# 2018-04-20 10:57:57 WARN  BLAS:61 - Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
# 2018-04-20 10:57:57 WARN  BLAS:61 - Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
>>> centers = kmodel.clusterCenters()
>>> centers
# [array([ 35.88461538,  31.46153846,  34.42307692]), array([ 5.12,  5.84,  4.84]), array([ 80.        ,  79.20833333,  78.29166667])]
