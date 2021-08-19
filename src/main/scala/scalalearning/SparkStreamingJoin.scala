package scalalearning

import org.apache.kafka.clients.producer._
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.{Seconds, State, StateSpec, StreamingContext}

import java.util.Properties

object SparkStreamingJoin extends App {

//  data schema

/*  stream has --- id, value   like 1:23  ..... 2:34 ....3:45

  static csv has --- name, id .....1:abc...2:def....3:ghi...

  we have to send ... name, value ......abc,23....def,34....ghi,45

*/

//  2 cases classes , Employee reads static data from csv, output returns output data

  case class Employee(name: String, empId: String)
  case class Output(name: String, value: String)


// creating spark session variables

  val spark: SparkSession = SparkSession.builder().getOrCreate()
  val ssc: StreamingContext = new StreamingContext(spark.sparkContext, Seconds(5))
  import spark.implicits._

  ssc.checkpoint("hdfs path")

//  read static data from csv..
  def getCsvData(path: String): Map[String, String] = {
    spark.read.csv(path).as[Employee].collect().foldLeft(Map[String,String]()) {
      case (accumulator: Map[String,String], value: Employee) => accumulator.+(value.empId -> value.name)
     }
  }

//  function to get name from id as state data and update stream data
  def stateUpdateFunction(key: String, data: Option[String], stateData: State[Map[String,String]]): Option[Output] = {

    val currentSession: Map[String, String] = if (stateData.exists()) {
      stateData.get()
    } else {
      getCsvData("path")
    }
    val dataArr: Array[String] = data.getOrElse("").split(":")
    val id: String = dataArr(0)
    val value: String = dataArr(1)

    currentSession.get(id).map(Output(_, value))
  }

//  kafka params..
  private val kafkaParams: Map[String, Object] = Map[String, Object](
    "bootstrap.servers" -> "localhost:9092",
    "key.deserializer" -> classOf[StringDeserializer],
    "value.deserializer" -> classOf[StringDeserializer],
    "auto.offset.reset" -> "latest",
    "enable.auto.commit" -> (false: java.lang.Boolean)
  )

//  reading input kafka data
  val topics: Array[String] = Array("topicA")
  val kafkaStream: DStream[(String, String)] = KafkaUtils.createDirectStream[String, String](
    ssc,
    PreferConsistent,
    Subscribe[String, String](topics, kafkaParams)
  )
    .map(record => (record.key, record.value))

//  processing kafka stream
  val processedData: DStream[Output] = kafkaStream
    .mapWithState(StateSpec.function(stateUpdateFunction _))
    .flatMap(x => x)


//  saving streaming data
  processedData.foreachRDD { rdd =>
    rdd.foreachPartition { iter =>
      val producer = createKafkaProducer();
      while (iter.hasNext) {
        sendToKafka(producer, iter.next())
      }
    }
  }

//  creating producer
  private def createKafkaProducer(): KafkaProducer[String, String] = {
    val props = new Properties();
    props.put("bootstrap.servers", "localhost:9092")
    props.put("key.deserializer", classOf[StringDeserializer])
    props.put("value.deserializer", classOf[StringDeserializer])
    props.put("topic","topicB")
    new KafkaProducer[String,String](props);
  }

//  sending data back to kafka
  private def sendToKafka(producer: KafkaProducer[String,String], op: Output): Unit = {
    producer.send(new ProducerRecord[String,String](op.name,op.value))
  }

}
