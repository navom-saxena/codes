from pyspark.sql import SparkSession


class count_words:
    if __name__ == '__main__':
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        lines = sc.textFile("C:/Users/aishwarya.dulwani/Desktop/kafkatest.txt")
        counts = lines.flatMap(lambda s: s.split(" ")).map(lambda a: (a, 1)).reduceByKey(lambda a, b: a + b)

        print(counts.take(10))
