package utils

import java.text.SimpleDateFormat
import java.time.temporal.ChronoUnit
import java.time.{Instant, LocalDate, ZoneId}
import java.util.{Calendar, Date, TimeZone}

import org.joda.time.DateTimeZone
import org.joda.time.format.DateTimeFormat

import scala.collection.immutable
import scala.collection.mutable.ListBuffer

object DateUtils {

  def generateBetweenDates(start: String, end: String): immutable.Seq[String] = {
    var startDate: LocalDate = LocalDate.parse(start)
    val endDate: LocalDate = LocalDate.parse(end)
    val totalDates: ListBuffer[String] = new ListBuffer[String]()

    while (startDate.isBefore(endDate)) {
      totalDates.append(startDate.toString)
      startDate = startDate.plusDays(1)
    }
    totalDates.toList
  }

  def endAndStartMilliSeconds(endDateOpt: Option[String], sinceDays: Int, timeZone: String): (Long, Long) = {
    val endDate: Instant = endDateOpt match {
      case Some(endDateStr) if endDateStr.nonEmpty =>
        LocalDate.parse(endDateStr).atStartOfDay(ZoneId.of(timeZone)).toInstant
      case _ => Instant.now()
    }
    val fromDate: Instant = endDate.minus(sinceDays, ChronoUnit.DAYS)
    val endTimeMilli: Long = endDate.toEpochMilli
    val startTimeMilli: Long = fromDate.toEpochMilli
    (endTimeMilli, startTimeMilli)
  }

  def generatePreviousDates(dt: String, offsetValue: Int): String = {
    val dateAux: Calendar = Calendar.getInstance()
    dateAux.setTime(new SimpleDateFormat("yyyy-MM-dd").parse(dt))
    dateAux.add(Calendar.DATE, -offsetValue)
    new SimpleDateFormat("yyyy-MM-dd").format(dateAux.getTime)
  }

  def getIstDateFromTimestamp(timestamp: Long): String = {
    val date: Date = new java.util.Date(timestamp)
    val sdf: SimpleDateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd")
    sdf.setTimeZone(java.util.TimeZone.getTimeZone("IST"))
    sdf.format(date)
  }

  def getStartEndTimestamp(inputDate: String, timeZone: String = "UTC"): Map[String, String] = {
    DateTimeZone.setDefault(DateTimeZone.forID(timeZone))
    TimeZone.setDefault(TimeZone.getTimeZone(timeZone))

    val DATE_FORMAT = "yyyy-MM-dd"
    val today = new SimpleDateFormat(DATE_FORMAT).parse(inputDate)
    val nextDay = new Date(today.getTime + (1000 * 60 * 60 * 24))

    Map("greaterThanTime" -> today.toInstant.toEpochMilli.toString, "lessThanTime" -> nextDay.toInstant.toEpochMilli.toString)
  }

  def timestampToGmtDt(timestampString: String): String = {
    timestampToDateString(timestampString)
  }

  def timestampToDateString(epochMillis: String,
                            pattern: String = "YYYY-MM-dd HH:mm:ss",
                            zone: String = "GMT"): String = {
    try {
      val epochMillisVal: Long = epochMillis.toLong
      DateTimeFormat
        .forPattern(pattern)
        .withZone(DateTimeZone.forID(zone)).print(epochMillisVal)
    } catch {
      case _: NumberFormatException => throw new NumberFormatException
    }
  }

}
