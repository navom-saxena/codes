package utils

import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}
import org.apache.spark.sql.functions._

object SsmSiteActivationMonthly {

  val spark = SparkSession.builder().getOrCreate()

  def processSsmSiteActivationMonthly(implicit spark: SparkSession) : Unit = {

    import spark.implicits._

    // put these 3 in import table trait

    val dDate: Dataset[Row] = spark.table("analysis_presentation.d_date").alias("dd")
    val saSites: Dataset[Row] = spark.table("prc_presentation.sa_sites").alias("ss")
    val smmStudyCountry: DataFrame = spark.table("cdm_views_tst.ssm_study_country")


    val bl: Dataset[Row] = dDate.join(saSites, $"ss.activation_baseline_date" === $"dd.iso_date")
      .groupBy("ss.project_code", "ss.country_code_3", "ss.country_name", "dd.year", "dd.month_short_desc")
      .agg(sum(when($"ss.activation_planned_date".isNull, 1).otherwise(0)).alias("baseline"))
      .select($"ss.project_code",
        $"ss.country_code_3",
        $"ss.country_name",
        $"dd.year",
        $"dd.month_short_desc".alias("month"),
        $"baseline")
      .alias("bl")

    val pl: Dataset[Row] = dDate.join(saSites, $"ss.activation_planned_date" === $"dd.iso_date")
      .groupBy("ss.project_code", "ss.country_code_3", "ss.country_name", "dd.year", "dd.month_short_desc")
      .agg(sum(when($"ss.activation_planned_date".isNull, 1).otherwise(0)).alias("planned"))
      .select($"ss.project_code",
        $"ss.country_code_3",
        $"ss.country_name",
        $"dd.year",
        $"dd.month_short_desc".alias("month"),
        $"planned")
      .alias("pl")

    val ac: Dataset[Row] = dDate.join(saSites, $"ss.activation_date" === $"dd.iso_date")
      .groupBy("ss.project_code", "ss.country_code_3", "ss.country_name", "dd.year", "dd.month_short_desc")
      .agg(sum(when($"ss.activation_planned_date".isNull, 1).otherwise(0)).alias("actual"))
      .select($"ss.project_code",
        $"ss.country_code_3",
        $"ss.country_name",
        $"dd.year",
        $"dd.month_short_desc".alias("month"),
        $"actual")
      .alias("ac")

    val mm: Dataset[Row] = dDate.select($"year",
      $"dd.month_short_desc".alias("month")).distinct()

    val allSct: Dataset[Row] = smmStudyCountry.select($"project_code",
      $"country_code3",
      $"country_name")
      .distinct()
      .alias("als")

    val ssmSiteActivationMonthly: DataFrame = allSct
      .join(mm)
      .join(bl, $"bl.project_code" === $"als.project_code"
        and $"bl.country_code_3" === $"als.country_code3"
        and $"bl.year" === $"mm.year"
        and $"bl.month" === $"mm.month", "left")
      .join(pl, $"pl.project_code" === $"als.project_code" and $"pl.country_code_3" === $"als.country_code3"
        and $"pl.year" === $"mm.year"
        and $"pl.month" === $"mm.month", "left")
      .join(ac, $"ac.project_code" === $"als.project_code"
        and $"ac.country_code_3" === $"als.country_code3"
        and $"ac.year" === $"mm.year"
        and $"ac.month" === $"mm.month", "left")
      .select(
        $"als.project_code"
        , $"als.country_code3"
        , $"als.country_name"
        , $"mm.year"
        , $"mm.month"
        , $"bl.baseline"
        , $"pl.planned"
        , $"ac.actual",
        lit("Oracle Site Activate")
      )

    ssmSiteActivationMonthly
      .write
      .mode(SaveMode.Overwrite)
      .insertInto("cdm_views_tst.ssm_site_activation_monthly")

  }

}
