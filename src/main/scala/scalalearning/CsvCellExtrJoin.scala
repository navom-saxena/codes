package scalalearning

import org.apache.spark.sql.{DataFrame, Dataset, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

object CsvCellExtrJoin {

  case class RowData(project_code: String, col: String, rownum : String)
  case class ProjectCodeProtocolId(project_code: String, value: String)

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val study: DataFrame = spark.table("cdm.study")
    val iceBudget: DataFrame = spark.table("ice.budget")
      .filter($"sheet_name" === lit("parameter"))
      .select("project_code", "value", "col", "rownum")
      .persist(StorageLevel.MEMORY_AND_DISK)

    val projectCodeProtocolId: Array[String] = study
      .filter($"protocol_id".isNaN)
      .join(iceBudget, study("project_code") === iceBudget("project_code")
        && lower(iceBudget("value")) rlike "protocol/studynumber")
      .select(study("project_code"), $"col", $"rownum").as[RowData]
      .collect()
      .map { rowData =>
        import rowData._
        val newCol: Char = col.toCharArray.headOption.map(x => (x + 1).toChar).getOrElse(' ')
        s"project_code = '$project_code' and rownum = '$rownum' and col = '$newCol'"
      }

    val sqlQueries: Array[ProjectCodeProtocolId] = projectCodeProtocolId.flatMap {
      iceBudget.filter(_).select("project_code", "value").as[ProjectCodeProtocolId].collect()
    }

    val studyWithProtocolId: DataFrame =
      study
        .join(spark.sparkContext.parallelize(sqlQueries).toDS(), Seq("project_code"), "left")
        .withColumn("protocol_id", when($"protocol_id".isNull, $"value").otherwise($"protocol_id"))

    studyWithProtocolId.write.mode(SaveMode.Append).format("orc").insertInto("new tablename")
  }

}
