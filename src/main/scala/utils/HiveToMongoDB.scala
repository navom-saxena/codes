package utils

import java.util.Date
import java.util.concurrent.TimeUnit

import com.mongodb.client.model.Projections
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.bson.BsonArray
import org.mongodb.scala.model.Filters.{and, equal}
import org.mongodb.scala.model.FindOneAndUpdateOptions
import org.mongodb.scala.model.Updates.{combine, set}
import org.mongodb.scala.{Document, MongoClient, MongoCollection, MongoDatabase, Observable}
import play.api.libs.json.{JsArray, JsObject, JsValue, Json}

import scala.concurrent.Await
import scala.concurrent.duration.Duration
import scala.util.Try

object HiveToMongoDB extends App {

  implicit class DocumentObservable[C](val observable: Observable[Document]) extends ImplicitObservable[Document] {
    override val converter: (Document) => String = (doc) => doc.toJson
  }

  implicit class GenericObservable[C](val observable: Observable[C]) extends ImplicitObservable[C] {
    override val converter: (C) => String = (doc) => doc.toString
  }

  val spark: SparkSession = SparkSession.builder().getOrCreate()
  val mongoUrl: String = "mongoUrl"
  val mongoDatabase: String = "mongoDatabase"
  val mongoCollection: String = "mongoCollection"
  val sourceTables: String = "sourceTables"
  val sourceDatabase: String = "sourceDatabase"
  val userName: String = "userName"
  val client: MongoClient = MongoClient(mongoUrl)
  val getMongoDatabase: MongoDatabase = client.getDatabase(mongoDatabase)
  val collection: MongoCollection[Document] = getMongoDatabase.getCollection(mongoCollection)
  val data: DataFrame = spark.read.option("header", "true").csv("someFile.txt")

  def findAndUpdateMongoData(inputDataframe: DataFrame): Unit = {
    val lis: Seq[Document] = inputDataframe.toJSON.collect().toSeq.map(x => Document(x))
    lis.map { details =>
      val col5Records: Seq[JsValue] = collection.find(and(equal("col1", details("col1")),
        equal("col2", details("col2"))))
        .projection(Projections.fields(Projections.include("col3"),
          Projections.excludeId())).results()
        .map(item => Json.parse(item.toJson()).as[JsValue])

      val col5Array: JsArray = col5Records.headOption.fold(JsArray()) { record =>
        (record \ "col3").as[List[JsValue]].foldLeft(JsArray()) { (currentDocument, json) =>
          currentDocument :+ json
        }
      }

      val username: JsValue = Json.parse(s"""{"created_by": "$userName"}""")
      val updatedCol5Value: Int = Try(details("col5").asInt32().getValue).getOrElse(details("col5").asString().getValue.toInt)

      val newCol5s = Json.obj("created_on" -> Json.obj("$date" -> new Date())) ++
        username.as[JsObject] ++ Json.obj("value" -> updatedCol5Value)

      val updatedArray: JsValue = col5Array :+ newCol5s

      collection.findOneAndUpdate(and(equal("col1", details("col1")),
        equal("col2", details("col2")),
        equal("subType", details("subType"))),
        combine(set("col4", updatedCol5Value),
          set("col5s",
            BsonArray.parse(Json.stringify(updatedArray)))),
        new FindOneAndUpdateOptions().upsert(true)
      ).results()
    }
  }

  trait ImplicitObservable[C] {
    val observable: Observable[C]
    val converter: (C) => String

    def printResults(initial: Option[String] = None): Unit = {
      initial.foreach(println)
      results().foreach(res => println(converter(res)))
    }

    def results(): Seq[C] = Await.result(observable.toFuture(), Duration(10, TimeUnit.SECONDS))

    def printHeadResult(initial: Option[String] = None): Unit = initial.foreach(x => println(s"$x${converter(headResult())}"))

    def headResult(): C = Await.result(observable.head(), Duration(10, TimeUnit.SECONDS))
  }

  findAndUpdateMongoData(data)

}
