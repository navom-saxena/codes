package scalalearning

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.dstream.{DStream, ReceiverInputDStream}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable
import scala.collection.immutable.List

// testing spark functions and understanding map, mapPartitions,cartesian.

object SparkFun extends App {

  val sparkConf: SparkConf = new SparkConf().setMaster("local[2]").setAppName("NetworkWordCount")
  val sparkSession: SparkSession = SparkSession.builder().config(sparkConf).getOrCreate()
  val sc: SparkContext = sparkSession.sparkContext
  val ssc = new StreamingContext(sc, Seconds(5))

  //  testing map partitions

  val list0: immutable.Seq[Int] = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  val rdd1: RDD[Int] = sc.parallelize(list0, 3)
  rdd1.mapPartitions { eachPartition =>
    eachPartition.foreach(print)
    println()
    eachPartition
  }.count()

  println("end of map partitions test --------------------")

  //  test map to see difference with map partitions

  val rdd2: Long = sc.parallelize(list0, 3).map { x =>
    print(x)
    x
  }.count()

  val list1: immutable.Seq[Int] = List(1, 2, 3, 4, 5)
  val rdd3: RDD[(Int, String)] = sc.parallelize(Seq((1, "hi"), (2, "yu"), (3, "io"), (40, "rr"), (58, "ret")))
  val rdd4: RDD[(Int, String)] = sc.parallelize(Seq((1, "yur"), (27, "wer"), (3, "rtyu"), (48, "werf"), (51, "ouptig")))
  val rdd5: RDD[((Int, String), (Int, String))] = rdd3.cartesian(rdd4)
  val lines: ReceiverInputDStream[String] = ssc.socketTextStream("localhost", 9999)

  /*testing partitioning of spark rdd with various length sequences*/

  while (n > 0) {
    val a0: Seq[Int] = (1 to 100).toList
    val a: RDD[Int] = sc.parallelize(a0, 15)
    val b: Array[Int] = a.take(1)
    println(b.length)
    println(a.getNumPartitions)
    println(a.partitions.length)
    println(sc.defaultParallelism)
    n = n - 1
  }

  rdd5.collect().foreach(println)

  println("end of partitioning test --------------------")

  // random spark streaming testing
  //  nc -lk 9999 and send data
  val words: DStream[String] = lines.flatMap(_.split(" "))
  val pairs: DStream[(String, Int)] = words.map(word => (word, 1))
  val wordCounts: DStream[(String, Int)] = pairs.reduceByKey(_ + _)
  var n = 5
  wordCounts.print()

  ssc.start()
  ssc.awaitTermination()

}
