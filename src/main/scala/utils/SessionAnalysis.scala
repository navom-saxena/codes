package utils

import java.time.Instant

import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.current_date
import org.apache.spark.sql.{DataFrame, Dataset, SaveMode, SparkSession}

import scala.collection.AbstractIterator

object SessionAnalysis extends App {

  case class HiveData(timestamp: String, userId: String) {
    val time: Long = Instant.parse(timestamp).getEpochSecond
  }

  case class Session(userId: String, firstTimestamp: Long, lastTimestamp: Long, count: Long)

  val spark: SparkSession = SparkSession.builder().master("local").getOrCreate()
  val sc: SparkContext = spark.sparkContext

  import spark.implicits._

  val maxSessionDuration: Long = 2 * 60 * 60L

  val input: Dataset[HiveData] = spark.sql("select * from table").as[HiveData]
    .repartition($"userId")
    .sortWithinPartitions("userId", "timestamp")

  input.mapPartitions(aggregateClicks(maxSessionDuration)).withColumn("date", current_date).write.mode(SaveMode.Append).insertInto("sessions")

  def aggregateClicks(maxSessionDuration: Long)(clicks: Iterator[HiveData]): Iterator[Session] =
    new SessionIterator(maxSessionDuration, clicks)

  class SessionIterator(maxSessionDuration: Long, rawClicks: Iterator[HiveData]) extends AbstractIterator[Session] with Iterator[Session] {

    val clicks: BufferedIterator[HiveData] = rawClicks.buffered

    var nextSession: Session = _

    override def hasNext: Boolean = {
      if (nextSession == null)
        nextSession = updateAndGetNextSession()

      nextSession != null
    }

    override def next(): Session = {
      val result: Session = {
        if (nextSession == null) updateAndGetNextSession()
        else try nextSession finally nextSession = null
      }
      if (result == null) Iterator.empty.next()
      else result
    }

    private def updateAndGetNextSession(): Session =
      if (!clicks.hasNext) {
        null
      } else {
        val first: HiveData = clicks.next()
        var last: HiveData = first
        var count: Int = 1

        while (clicks.hasNext && inSameSession(maxSessionDuration)(last, clicks.head)) {
          last = clicks.next()
          count = count + 1
        }

        Session(first.userId, first.time, last.time, count)
      }

    private def inSameSession(maxSessionDuration: Long)(c1: HiveData, c2: HiveData): Boolean =
      c1.userId == c2.userId && Math.abs(c1.time - c2.time) < maxSessionDuration

  }

  // create table sessions(userid string, first_timestamp BigInt, last_timestamp BigInt, count int) partitioned by (date string)

  // create table SessionsCount(total_sessions BigInt) stored as orc
  // create table daily_user_time_spent_mins(time String) stored as orc
  // create table monthly_user_time_spent_mins(time String) stored as orc

  val df1: DataFrame = spark.sql("select distinct(first_timestamp,last_timestamp) as sessions_count from sessions where date = CURRENT_DATE")
  val df2: DataFrame = spark.sql("select (sum((last_timestamp - first_timestamp) as diff) / 60) as total_time_min from sessions group by userid having date = CURRENT_DATE")
  val df3: DataFrame = spark.sql("select (sum((last_timestamp - first_timestamp) as diff) / 60) as total_time_min from sessions group by userid having date > cast(date_sub(CURRENT_DATE, 30) as string)")

  df1.write.mode(SaveMode.Overwrite).format("orc").insertInto("SessionsCount")
  df2.write.mode(SaveMode.Overwrite).format("orc").insertInto("daily_user_time_spent_mins")
  df3.write.mode(SaveMode.Overwrite).format("orc").insertInto("monthly_user_time_spent_mins")

}
