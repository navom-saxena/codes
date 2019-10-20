from pyspark.sql import SparkSession


class Ratings:
    if __name__ == '__main__':
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext

        rating = sc.textFile("/Users/navom_saxena/Downloads/ratings1.csv")
        ratingrdd = rating.map(lambda t: (t.split(",")[0], t.split(",")[2]))
        finalratingrdd = ratingrdd.groupByKey()
        max = 0.0
        min = 0.0
        sum = 0.0
        c = 0
        for (k, v) in finalratingrdd.collect():
            s = k + " " + str(v)
            for i in v:
                rating = float(i)
                if (max < rating):
                    max = rating
                if (min > rating):
                    min = rating
                sum += rating
                c = c + 1

            print("max......")
            print(max)
            print("min......")
            print(min)
            avg = sum / c
            print("avg.....")
            print(avg)
            print("userid......")
            print(k)

        # print(finalratingrdd.take(5))
