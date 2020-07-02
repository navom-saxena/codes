package utils

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object SqlToDfConversion {

  //  sql hive query to dataFrame conversion

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder().appName("PRC_Site_Activation")
      .config("spark.sql.warehouse.dir", "/apps/hive/warehouse")
      .config("hive.metastore.uris", "thrift://1088259-rsus-hdmatst01.incresearch.com:9083").enableHiveSupport().getOrCreate()

    import spark.implicits._

    val study_rag = spark.table("cdm.study_rag")
    val project_mapping = spark.table("cdm.project_mapping")
    val custom_data = spark.table("raw_pv_c.custom_data")
    val planning_entity = spark.table("raw_pv_c.planning_entity")
    val structure = spark.table("raw_pv_c.structure")

    val cm_index = spark.table("prc_presentation.cm_index")
    val pa_proj = spark.table("fdm_presentation.pa_proj_fin_all")
    val employee_role = spark.table("cdm.grm_employee_role")
    val emp = spark.table("analysis_presentation.network_users_all")


    val pm_study = spark.table(" prc_presentation.pm_study")
    val sa_index = spark.table("prc_presentation.sa_index")
    val se_v_prc_study_index = spark.table("prc_presentation.se_v_prc_study_index")
    val dm_global_details = spark.table("prc_presentation.dm_global_details")

    val clinical_project_v = spark.table("mdm.clinical_project_v")

    //  select d1.quarter year_quarter,
    //  count(distinct case when substring(iso_date, 1, 10) <=
    //  current_date then iso_date else null end) / count(distinct iso_date) quarter_so_far
    //  from analysis_presentation.d_date d1 group by d1.quarter
    //

    //    select d1.quarter as quarter,
    //    count(distinct case when substring(iso_date, 1, 10) <= current_date then iso_date else null end)
    //    /  count(distinct iso_date) as prorated_target
    //    from  analysis_presentation.d_date d1
    //    group by d1.quarter having
    //    count(distinct case when substring(iso_date, 1, 10) <= current_date then iso_date else null end)/count(distinct iso_date) !=0

    val Clinical_Study = custom_data.alias("cd").join(planning_entity.alias("pe"), "planning_code")
      .join(structure.alias("s"), $"pe.code22" === $"s.structure_code")
      .join(project_mapping.alias("pm"),
        regexp_replace(trim($"cd.inc_project_code"), "\\t", "") === $"pm.child_project_code"
          && lit("PV") === $"pm.child_source", "left")
      .filter($"s.description" === lit("Clinical/Functional Services")
        && ($"cd.inc_project_code" =!= lit("") || $"cd.inc_project_code".isNotNull))
      .select(coalesce($"pm.master_project_code",
        regexp_replace(trim($"cd.inc_project_code"), "\\t", "")).alias("project_code"))
      .distinct()

    val Clinical_Study_U = cm_index.alias("ci")
      .join(Clinical_Study, Clinical_Study("project_code") === regexp_replace($"ci.project_code", "U", ""))
      .filter(Clinical_Study("project_code").isNotNull && $"ci.project_code".like("%U"))
      .select(Clinical_Study("project_code").alias("project_code1"), $"ci.project_code").distinct()

    val Clinical_Study_All = Clinical_Study
      .select($"project_code", $"project_code".alias("project_code1"))
      .union(Clinical_Study_U.select($"project_code", $"project_code1"))

    val PV_Status = study_rag
      .withColumn("ROW_NUM", row_number.over(Window.partitionBy($"project_code", $"rag_type")
        .orderBy($"project_code", $"rag_type", $"last_modified_date".desc)))
      .filter($"rag_type" === lit("Project Status") && $"ROW_NUM" === 1)
      .select($"project_code", $"rag_value".alias("study_status"))

    val Direct_Budget = pa_proj
      .withColumn("row_num", row_number().over(Window.partitionBy("p_number")
        .orderBy(from_unixtime(unix_timestamp(upper($"mcal_period_name"), "MMM-dd"), "MMdd").desc)))
      .filter($"row_num" === 1)
      .select($"p_number", $"bud_rev")
      .groupBy($"p_number").agg(sum($"bud_rev").alias("direct_budget"))
      .select($"p_number".alias("project_Code"), $"direct_budget")

    val functionalLeadSubQ1 = employee_role.alias("er").join(emp.alias("emp"),
      regexp_replace($"emp.emplid", "^0+(?!$)", "")
        === regexp_replace($"er.employee_id", "^0+(?!$)", ""), "left")
      .filter($"er.proj_role".isin("SSUPDL", "SIL", "PDM", "LCRA", "LCRA-UB")
        && $"er.lead_indicator" === "Y" && $"er.assign_sts" === "A")
      .select(
        $"er.project_code",
        when($"er.project_code".isin("SSUPDL"), "SSUPDL")
          .when($"er.project_code".isin("SIL"), "SIL")
          .when($"er.project_code".isin("PDM"), "PDM")
          .when($"er.project_code".isin("LCRA", "LCRA-UB"), "LCRA")
          .otherwise($"er.proj_role").alias("proj_role"),
        $"er.employee_id", $"er.lead_indicator",
        concat($"emp.last_name", lit(", "), $"emp.first_name", lit(" "), $"emp.middle_name")
          .alias("employee_name"),
        to_date($"er.start_date").alias("start_date"),
        to_date($"er.end_date").alias("end_date")
      ).distinct().alias("ER")


    val functionalLeadSubQ2 = employee_role.filter($"proj_role".isin("SSUPDL", "SIL", "PDM", "LCRA", "LCRA-UB")
      && $"lead_indicator" === "Y" && $"assign_sts" === "A"
      && current_date.between(to_date($"start_date"), to_date($"end_date")))
      .groupBy($"project_code",
        when($"proj_role".isin("SSUPDL"), "SSUPDL")
          .when($"proj_role".isin("SIL"), "SIL")
          .when($"proj_role".isin("PDM"), "PDM")
          .when($"proj_role".isin("LCRA", "LCRA-UB"), "LCRA")
          .otherwise($"proj_role").alias("proj_role")).agg(min(to_date($"start_date")).alias("start_date"))
      .select(
        $"project_code",
        when($"proj_role".isin("SSUPDL"), "SSUPDL")
          .when($"proj_role".isin("SIL"), "SIL")
          .when($"proj_role".isin("PDM"), "PDM")
          .when($"proj_role".isin("LCRA", "LCRA-UB"), "LCRA")
          .otherwise($"proj_role").alias("proj_role"), $"start_date").alias("R1")

    val Functional_Lead = functionalLeadSubQ1.join(functionalLeadSubQ2, $"ER.project_code" === $"R1.project_code"
      && to_date($"ER.start_date") === $"R1.start_date" && $"ER.proj_role" === $"R1.proj_role")
      .join(project_mapping.alias("pm"), $"pm.child_project_code" === $"er.project_code" && $"pm.child_source" === lit("GRM"), "left")
      .filter($"ER.lead_indicator" === "Y" && current_date.between(to_date($"ER.start_date"), to_date($"ER.end_date")))
      .groupBy(coalesce($"pm.master_project_code", $"ER.project_code").alias("project_code"))
      .agg(max(
        when($"ER.proj_role".isin("SSUPDL"), col("ER.employee_name")).otherwise(null)).alias("SSU_PDL_Name"),
        max(when($"ER.proj_role".isin("SIL"), col("ER.employee_name")).otherwise(null)).alias("Site_ID_Name"),
        max(when($"ER.proj_role".isin("PDM"), col("ER.employee_name")).otherwise(null)).alias("PDM_Name"),
        max(when($"ER.proj_role".isin("LCRA", "LCRA-UB"), col("er.employee_name")).otherwise(null)).alias("LCRA_Name"))
      .select($"project_code", $"SSU_PDL_Name", $"Site_ID_Name", $"PDM_Name", $"LCRA_Name")
    val clinical_project_v_picked_one = clinical_project_v
      .filter($"exp_flag" === "true" && $"exclude_from_reports" === "false")
      .groupBy($"project_code").agg(count($"*")
      .alias("cnt")).filter($"cnt" === 1).select($"project_code")

    val picked_one = clinical_project_v.alias("s")
      .join(clinical_project_v_picked_one.alias("b")
        , $"s.project_code" === $"b.project_code").filter((($"s.project_code").isNotNull && $"s.project_code" != "")
      && $"s.exp_flag" === "true" && $"s.exclude_from_reports" === "false")
      .select($"s.hrow_id",
        $"s.int_project_id",
        $"s.project_code",
        $"s.protocol_id",
        $"s.protocol_title",
        $"s.project_name".alias("study_name"),
        $"s.int_sponsor_id",
        $"s.sponsor_name",
        $"s.sponsor_parent_company_name",
        $"s.owning_business_unit",
        $"s.bu_code",
        $"s.owning_business_unit_sub_group",
        $"s.study_phase",
        $"s.project_stage_name",
        $"s.drug_device_name",
        $"s.therapeutic_area",
        $"s.primary_indication".alias("indication"),
        $"s.primary_sub_indication",
        $"s.source",
        $"s.isdeleted",
        $"s.last_modified_date",
        $"s.h_last_processed_date",
        $"s.last_processed_date")

    val Single_record = clinical_project_v
      .withColumn("ROW_NUM", row_number.over(Window.partitionBy($"project_code")
        .orderBy($"hrow_id".asc))).filter(($"project_code").isNotNull && $"project_code" != "")
      .select($"hrow_id",
        $"int_project_id",
        $"project_code",
        $"protocol_id",
        $"protocol_title",
        $"project_name".alias("study_name"),
        $"int_sponsor_id",
        $"sponsor_name",
        $"sponsor_parent_company_name",
        $"owning_business_unit",
        $"bu_code",
        $"owning_business_unit_sub_group",
        $"study_phase",
        $"project_stage_name",
        $"drug_device_name",
        $"therapeutic_area",
        $"primary_indication".alias("indication"),
        $"primary_sub_indication",
        $"source",
        $"isdeleted",
        $"last_modified_date",
        $"h_last_processed_date",
        $"last_processed_date",
        $"ROW_NUM")

    val first_record = Single_record.filter($"ROW_NUM" === 1).select($"hrow_id",
      $"int_project_id",
      $"project_code",
      $"protocol_id",
      $"protocol_title",
      $"study_name",
      $"int_sponsor_id",
      $"sponsor_name",
      $"sponsor_parent_company_name",
      $"owning_business_unit",
      $"bu_code",
      $"owning_business_unit_sub_group",
      $"study_phase",
      $"project_stage_name",
      $"drug_device_name",
      $"therapeutic_area",
      $"indication",
      $"primary_sub_indication",
      $"source",
      $"isdeleted",
      $"last_modified_date",
      $"h_last_processed_date",
      $"last_processed_date")

    val ONCE = spark.table("once_table")
    val project_code_union = ONCE.union(picked_one).select($"project_code")

    val sf = ONCE.union(picked_one.alias("pone").join(ONCE.alias("opc"), $"pone.project_code" === $"opc.project_code", "left")
      .filter($"opc.project_code".isNull).selectExpr("pone.*")).union(first_record.alias("frec").join(project_code_union.alias("pcu")
      , $"frec.project_code" === $"pcu.project_code", "left").filter($"pcu.project_code".isNull).selectExpr("frec.*"))

    val cm_index_pcode = cm_index.select("project_code")
    val pm_study_pcode = pm_study.select("project_code")
    val sa_index_pcode = sa_index.select("project_code")
    val se_v_prc_study_index_pcode = se_v_prc_study_index.select("project_code")
    val dm_global_detailsp_code = dm_global_details.filter($"site_ref_id".isNotNull).selectExpr("project_id")

    val All_Projects = cm_index_pcode.union(pm_study_pcode).union(sa_index_pcode).union(se_v_prc_study_index_pcode).union(dm_global_detailsp_code).distinct()

    val cm_index_x = cm_index.selectExpr("project_code", "site_number", "country_code3", "country_name", "region")
    val sa_index_x = sa_index.selectExpr("project_code", "site_number", "country_code3", "country_name", "region")
    val dm_global_details_x = dm_global_details.selectExpr("project_id", "site_ref_id", "country_code3", "country_name", "edc_region")

    val All_Sites = cm_index_x.union(sa_index_x).union(dm_global_details_x).selectExpr("project_code", "site_number", "country_code3", "country_name", "region").distinct()
    val pm = pm_study.select("project_code", "study_name", "int_study_id", "in_enrollment", "milestone_actual_date", "recent_milestone_name", "pd_name", "pl_assessment", "pl_assessment_comments", "pl_assessment_date", "pl_name").distinct()

    val sa = sa_index.selectExpr("project_code", "site_number", "site_status").distinct()

    val se = se_v_prc_study_index.groupBy("project_code", "int_study_id", "country_code_3", "in_enrollment", "indication", "last_modified_date").agg(sum("sites_budgeted").as("sites_budgeted")).selectExpr("project_code", "int_study_id", "country_code_3", "cast (in_enrollment as string) in_enrollment", "indication", "last_modified_date", "sites_budgeted")

    val dm = dm_global_details.filter($"site_ref_id".isNotNull && $"Subject_name".isNotNull).groupBy("project_id", "site_ref_id", "Subject_name").agg(max("standard_subject_status").as("standard_subject_status"), max("subject_status").as("subject_status")).selectExpr("project_id as project_code", "site_ref_id as site_number", "Subject_name as Subject_name", "standard_subject_status", "subject_status")

    val dmt = dm_global_details.filter($"site_ref_id".isNotNull && $"Subject_name".isNotNull).groupBy("project_id").agg(max("dm_director").as("dm_director"), max("dm_manager").as("dm_manager"), max("dm_study_flag").as("dm_study_flag"), max("lcd_manager").as("lcd_manager"), max("lcd_region").as("lcd_region"), max("project_Lead").as("project_Lead"), max("protocol_id").as("protocol_id"), max("protocol_title").as("protocol_title")).selectExpr("project_id as project_code", "dm_director", "dm_manager", "dm_study_flag", "lcd_manager", "lcd_region", "project_Lead", "protocol_id", "protocol_title")

    val All_PRC = All_Projects.alias("all_projects").join(All_Sites.alias("all_sites"), $"all_projects.project_code" === $"all_sites.project_code", "left").join(pm.alias("pm"), $"pm.project_code" === $"all_projects.project_code", "left").join(sa.alias("sa"), $"sa.project_code" === $"all_sites.project_code" && $"sa.site_number" === $"all_sites.site_number", "left").join(se.alias("se"), $"se.project_code" === $"all_sites.project_code" && $"se.country_code_3" === $"all_sites.country_code3", "left").join(dm.alias("dm"), $"dm.project_code" === $"all_sites.project_code" && $"dm.site_number" === $"all_sites.site_number", "left").join(dmt.alias("dmt"), $"dmt.project_code" === $"all_sites.project_code", "left").selectExpr("all_projects.project_code", "all_sites.site_number", "all_sites.country_code3", "all_sites.country_name", "all_sites.region", "sa.site_status", "se.sites_budgeted", "dm.standard_subject_status as standard_subject_status", "dm.subject_name as subject_name", "dm.subject_status as subject_status", "pm.study_name", "COALESCE( pm.int_study_id,se.int_study_id ) as int_study_id", "dmt.dm_director dm_director", "dmt.dm_manager dm_manager", "dmt.dm_study_flag dm_study_flag", "COALESCE( pm.in_enrollment,se.in_enrollment ) as in_enrollment", "se.indication", "dmt.lcd_manager lcd_manager", "dmt.lcd_region lcd_region", "pm.milestone_actual_date", "pm.recent_milestone_name", "pm.pd_name", "pm.pl_assessment", "pm.pl_assessment_comments", "pm.pl_assessment_date", "COALESCE(pm.pl_name, dmt.project_lead) project_Lead", "se.last_modified_date").distinct()

  }

  // sql hive query

  /*

    SELECT   * -- count (distinct project_code) , count(*)
    FROM (
      WITH Clinical_Study AS -- Blinded Projects e.g 3104
    (
      select
        distinct  COALESCE(pm.master_project_code, regexp_replace(trim(cd.inc_project_code),"\\t","") ) project_code
        from
        raw_pv_c.custom_data cd
      join    raw_pv_c.planning_entity pe
      on (cd.planning_code = pe.planning_code)
    join    raw_pv_c.structure s
      on (pe.code22 = s.structure_code)
    left join    cdm.project_mapping pm
      on pm.child_project_code = regexp_replace(trim(cd.inc_project_code),"\\t","")
    and pm.child_source = 'PV'
    where
    s.description = 'Clinical/Functional Services'
    and (cd.inc_project_code <> '' or cd.inc_project_code is not NULL)
    ),
    Clinical_Study_U AS -- Un-blinded Studies e.g 171 Unblinded Studies
      (
        Select
          distinct ci.project_code, cs.project_code project_code1
        from
        prc_presentation.cm_index ci
      left join clinical_study cs on cs.project_code = regexp_replace(ci.project_code, "U", "")
    where
    ci.project_code like '%U'
    and cs.project_code is not null
    ),
    Clinical_Study_All AS -- 3104 Blinded Projects + 171 Unblinded Studies = Total 3275 Projects
    (
      select count(project_code), count(project_code) from (
      select
        distinct  COALESCE(pm.master_project_code, regexp_replace(trim(cd.inc_project_code),"\\t","") ) project_code
        from
        raw_pv_c.custom_data cd
      join    raw_pv_c.planning_entity pe
      on (cd.planning_code = pe.planning_code)
    join    raw_pv_c.structure s
      on (pe.code22 = s.structure_code)
    left join    cdm.project_mapping pm
      on pm.child_project_code = regexp_replace(trim(cd.inc_project_code),"\\t","")
    and pm.child_source = 'PV'
    where
    s.description = 'Clinical/Functional Services'
    and (cd.inc_project_code <> '' or cd.inc_project_code is not NULL)
    ) a
    UNION
    select project_code, project_code1 from (
      Select
        distinct ci.project_code, cs.project_code project_code1
      from
      prc_presentation.cm_index ci
      left join (
      select
        distinct  COALESCE(pm.master_project_code, regexp_replace(trim(cd.inc_project_code),"\\t","") ) project_code
        from
        raw_pv_c.custom_data cd
      join    raw_pv_c.planning_entity pe
      on (cd.planning_code = pe.planning_code)
    join    raw_pv_c.structure s
      on (pe.code22 = s.structure_code)
    left join    cdm.project_mapping pm
      on pm.child_project_code = regexp_replace(trim(cd.inc_project_code),"\\t","")
    and pm.child_source = 'PV'
    where
    s.description = 'Clinical/Functional Services'
    and (cd.inc_project_code <> '' or cd.inc_project_code is not NULL)
    ) cs on cs.project_code = regexp_replace(ci.project_code, "U", "")
    where
    ci.project_code like '%U'
    and cs.project_code is not null
    )
    ),
    PV_Status AS --identify the PV Project status
      (SELECT project_code, study_status FROM (
        SELECT r1.project_code ,
        r1.rag_value study_status,
        ROW_NUMBER() OVER(PARTITION BY project_code, rag_type
          ORDER BY  project_code, rag_type, last_modified_date DESC)  as ROW_NUM
          FROM   cdm.study_rag r1
          where rag_type = 'Project Status'
    ) xxx
    where row_num = 1),
    Direct_Budget AS --identify the  Project's Direct Budget
    (
      SELECT          p_number AS             project_Code
      , SUM (bud_rev)         direct_budget
      FROM    (
        SELECT
          p_number
        , bud_rev
        , ROW_NUMBER () OVER (PARTITION BY p_number ORDER BY from_unixtime (unix_timestamp (UPPER (mcal_period_name),"MMM-dd"),"MMdd") DESC) row_num
          FROM
          fdm_presentation.pa_proj_fin_all
    ) sub
      WHERE           row_num = 1
    GROUP BY        p_number
    --      -- Earlier Query
      --                        select project_id project_code, sum(global2_total_direct_budget_amt) direct_budget
      --                        from sbi_presentation.pd_project_fin_info
    --                        group by project_id
    ),
    Functional_Lead AS --identify the Functionals Lead based on the Role, use max to eliminate duplicates
    (
      select  -- Finding the Lead per Project_Role and avoiding duplicate if Leads
    COALESCE(pm.master_project_code, er.project_code )  project_code
    , MAX(Case when er.proj_role in ( 'SSUPDL'          ) Then er.employee_name  Else NULL end) as SSU_PDL_Name
    , MAX(Case when er.proj_role in ( 'SIL'             ) Then er.employee_name  Else NULL end) as Site_ID_Name
    , MAX(Case when er.proj_role in ( 'PDM'             ) Then er.employee_name  Else NULL end) as PDM_Name
    , MAX(Case when er.proj_role in ( 'LCRA'            ) Then er.employee_name  Else NULL end) as LCRA_Name
      from
    (-- 'ER' Getting the required attributes for Active Project Role Leads
      select distinct  er.project_code
    , Case       when er.proj_role in ( 'SSUPDL'          ) Then 'SSUPDL'
    when er.proj_role in ( 'SIL'             ) Then 'SIL'
    when er.proj_role in ( 'PDM'             ) Then 'PDM'
    when er.proj_role in ( 'LCRA', 'LCRA-UB' ) Then 'LCRA'
    Else er.proj_role  end proj_role
    , er.employee_id
    , er.lead_indicator
    , emp.last_name ||', '||emp.first_name||' '||emp.middle_name  employee_name
    , to_date(er.start_date) start_date
    , to_date(er.end_date) end_date
      from
    cdm.grm_employee_role  ER
      left join   analysis_presentation.network_users_all emp
      on  regexp_replace( emp.emplid, '^0+(?!$)', '') =  regexp_replace( er.employee_id, '^0+(?!$)', '')
    where
    er.proj_role in  ( 'SSUPDL','SIL','PDM','LCRA', 'LCRA-UB')
    and er.lead_indicator ='Y' and er.assign_sts ='A'
    )
    ER
    Join
    ( -- 'R1' Finding the Lead's least start_Date for a Project_role & Project
      select   er.project_code
    , Case       when er.proj_role in ( 'SSUPDL'          ) Then 'SSUPDL'
    when er.proj_role in ( 'SIL'             ) Then 'SIL'
    when er.proj_role in ( 'PDM'             ) Then 'PDM'
    when er.proj_role in ( 'LCRA', 'LCRA-UB' ) Then 'LCRA'
    Else er.proj_role  end proj_role
    , min(to_date(er.start_date)) start_date
      from   cdm.grm_employee_role  er
      where  er.proj_role in  ( 'SSUPDL','SIL','PDM','LCRA', 'LCRA-UB')
    and er.lead_indicator ='Y' and er.assign_sts ='A'
    and current_date BETWEEN  to_date(er.start_date) and to_date(er.end_date)
    Group by
      project_code
    , Case       when er.proj_role in ( 'SSUPDL'          ) Then 'SSUPDL'
    when er.proj_role in ( 'SIL'             ) Then 'SIL'
    when er.proj_role in ( 'PDM'             ) Then 'PDM'
    when er.proj_role in ( 'LCRA', 'LCRA-UB' ) Then 'LCRA'
    Else er.proj_role  end
    ) R1
    ON ER.project_code = R1.project_code and to_date(ER.start_date) = R1.start_date and er.proj_role= r1.proj_role
    Left join   cdm.project_mapping pm
      ON  pm.child_project_code = er.project_code and pm.child_source = 'GRM'
    where
    ER.lead_indicator ='Y' and  current_date BETWEEN  to_date(ER.start_date) and to_date(ER.end_date)
    Group By
      COALESCE(pm.master_project_code, er.project_code )
    ),
    ONCE AS         -- identify the  Project's which apprear only once
    (
      select
        hrow_id,
      int_project_id,
      project_code,
      protocol_id,
      protocol_title,
      project_name study_name,
      int_sponsor_id,
      sponsor_name,
      sponsor_parent_company_name,
      owning_business_unit,
      bu_code,
      owning_business_unit_sub_group,
      study_phase,
      project_stage_name,
      drug_device_name,
      therapeutic_area,
      primary_indication  indication,
      primary_sub_indication,
      source,
      isdeleted,
      last_modified_date,
      h_last_processed_date,
      last_processed_date
        from
        mdm.clinical_project_v
        where
        ( project_code <> '' and project_code is not null )
    and project_code in ( select project_code from mdm.clinical_project_v group by project_code having count(*) = 1)
    -- and project_stage_name in ( '5 – Award','6 – Contract')
    -- and to_date(createddate) > '2014-01-01'
    ) ,
    picked_one AS          -- identify the  Project's which apprear only once based on exp_flag = 'true' & exclude_from_reports = 'false' condition
    (
      select
        s.hrow_id,
    s.int_project_id,
    s.project_code,
    s.protocol_id,
    s.protocol_title,
    s.project_name study_name,
    s.int_sponsor_id,
    s.sponsor_name,
    s.sponsor_parent_company_name,
    s.owning_business_unit,
    s.bu_code,
    s.owning_business_unit_sub_group,
    s.study_phase,
    s.project_stage_name,
    s.drug_device_name,
    s.therapeutic_area,
    s.primary_indication  indication,
    s.primary_sub_indication,
    s.source,
    s.isdeleted,
    s.last_modified_date,
    s.h_last_processed_date,
    s.last_processed_date
    from
    mdm.clinical_project_v s
      where  ( s.project_code <> '' and s.project_code is not null )
    and s.exp_flag = 'true' and  s.exclude_from_reports = 'false'
    and s.project_code in ( select s.project_code from mdm.clinical_project_v s
      where exp_flag = 'true' and exclude_from_reports = 'false' group by project_code having count(*) = 1)
    --        and project_stage_name in ( '5 – Award','6 – Contract')
    --        and to_date(createddate) > '2014-01-01'
    ),
    first_record AS        -- Select first record from Multiple records based on hrow_id
      (
        select
          hrow_id,
        int_project_id,
        project_code,
        protocol_id,
        protocol_title,
        study_name,
        int_sponsor_id,
        sponsor_name,
        sponsor_parent_company_name,
        owning_business_unit,
        bu_code,
        owning_business_unit_sub_group,
        study_phase,
        project_stage_name,
        drug_device_name,
        therapeutic_area,
        indication,
        primary_sub_indication,
        source,
        isdeleted,
        last_modified_date,
        h_last_processed_date,
        last_processed_date
          from (
          select
            hrow_id,
          int_project_id,
          project_code,
          protocol_id,
          protocol_title,
          project_name study_name,
          int_sponsor_id,
          sponsor_name,
          sponsor_parent_company_name,
          owning_business_unit,
          bu_code,
          owning_business_unit_sub_group,
          study_phase,
          project_stage_name,
          drug_device_name,
          therapeutic_area,
          primary_indication indication,
          primary_sub_indication,
          source,
          isdeleted,
          last_modified_date,
          h_last_processed_date,
          last_processed_date,
          ROW_NUMBER() OVER(PARTITION BY project_code ORDER BY hrow_id )  as ROW_NUM
            from
            mdm.clinical_project_v
            where  ( project_code <> '' and project_code is not null )
    --        and project_stage_name in ( '5 – Award','6 – Contract')
    --        and to_date(createddate) > '2014-01-01'
    ) Single_record
    where   ROW_NUM = 1
    ),
    SF AS        -- identify of a Proper SalesForce Record
    (
      select * from
      (
        select * from once
          union
          select * from picked_one   where project_code not in ( select project_code from once )
          union
          select * from first_record where project_code not in ( select project_code from once union all  select project_code from picked_one)
      )  Distinct_Projects
      ),
    ALL_PRC AS       -- Consolidations of All PRC Poject's and available attributes
      (
        SELECT  Distinct
          all_projects.project_code
            -- Site Levels
        , all_sites.site_number
        , all_sites.country_code3
        , all_sites.country_name
        , all_sites.region
        , SA.site_status
        , SE.sites_budgeted
        -- Patient Levels
        , (DM.standard_subject_status) standard_subject_status
        , (DM.subject_name) subject_name
        , (DM.subject_status) subject_status
        -- Study related
        , PM.study_name
        , COALESCE( PM.int_study_id  , SE.int_study_id )        int_study_id
        , DMT.dm_director dm_director
        , DMT.dm_manager dm_manager
        , DMT.dm_study_flag dm_study_flag
        , COALESCE( PM.in_enrollment , SE.in_enrollment )       in_enrollment
        , SE.indication
        , DMT.lcd_manager lcd_manager
        , DMT.lcd_region lcd_region
        , PM.milestone_actual_date
        , PM.recent_milestone_name
        , PM.pd_name
        , PM.pl_assessment
        , PM.pl_assessment_comments
        , PM.pl_assessment_date
        , COALESCE( PM.pl_name, DMT.project_lead) project_Lead
        , SE.last_modified_date
        --, SE.isdeleted
        FROM
        (       -- Identify distinct  Projects
          Select
          project_code
          from
          prc_presentation.cm_index
        Union
        Select
        project_code
        from
        prc_presentation.pm_study
    Union
    Select
    project_code
    from
    prc_presentation.sa_index
    Union
    Select
    project_code
    from
    prc_presentation.se_v_prc_study_index
    Union
    Select
    project_id
    from
    prc_presentation.dm_global_details
    where
    site_ref_id is not null
    )
    All_Projects
    LEFT JOIN
      (       -- Identify distinct  Projects & Sites
        select
        project_code
        , site_number
        , country_code3
        , country_name
        , region
        from (
        Select project_code  , site_number , country_code3 , country_name , region
        from prc_presentation.cm_index
        Union
        Select project_code  , site_number , country_code3 , country_name , region
        from prc_presentation.sa_index
        Union
        Select project_id    , site_ref_id , country_code3 , country_name , edc_region
        from prc_presentation.dm_global_details
      ) x
        group by  project_code
        , site_number
        , country_code3
        , country_name
        , region

      )
    All_Sites
    ON      all_projects.project_code = all_sites.project_code
    Left JOIN
      (       -- Study Level
        Select    Distinct
        project_code
        , study_name --
        , int_study_id --
        , in_enrollment
        , milestone_actual_date
        , recent_milestone_name
        , pd_name
        , pl_assessment
        , pl_assessment_comments
        , pl_assessment_date
        , pl_name
        from
        prc_presentation.pm_study
      ) PM
      ON        PM.project_code = all_projects.project_code
    Left JOIN
      (       -- Study & Site Levels
        Select    Distinct
        project_code
        , site_number
        , site_status
        from
        prc_presentation.sa_index
      ) SA
      ON      SA.project_code = all_sites.project_code
    AND     SA.site_number  = all_sites.site_number
    Left JOIN
      (       -- Subject & Country Level
        Select
        project_code
        , int_study_id
        , country_code_3
        , cast (in_enrollment as string) in_enrollment
        , indication
        , last_modified_date
        , sum(sites_budgeted) sites_budgeted
        from
        prc_presentation.se_v_prc_study_index
    group by
      project_code
    , int_study_id
    , country_code_3
    , cast (in_enrollment as string)
    , indication
    , last_modified_date
    ) SE
    ON      SE.project_code   = all_sites.project_code
    AND     SE.country_code_3 = all_sites.country_code3
    Left JOIN
      (         -- Study , Site & Subject Levels
        Select
        project_id                    project_code
        , site_ref_id                   site_number
        , subject_name                  subject_name
        , MAX(standard_subject_status)  standard_subject_status
        , MAX(subject_status)           subject_status
        --, MAX(dm_director)              dm_director
        -- , MAX(dm_manager)               dm_manager
        --  , MAX(dm_study_flag)            dm_study_flag
        -- , MAX(lcd_manager)              lcd_manager
        -- , MAX(lcd_region)               lcd_region
        -- , MAX(project_Lead)             project_Lead
        -- , MAX(protocol_id)              protocol_id
        -- , MAX(protocol_title)           protocol_title
        from
        prc_presentation.dm_global_details
    where
    site_ref_id is NOT NULL and subject_name is NOT NULL
    group by
      project_id
    , site_ref_id
    , subject_name
    ) DM
    ON      DM.project_code = all_sites.project_code
    AND     DM.site_number  = all_sites.site_number
    LEFT JOIN (Select
      project_id                    project_code
      --    , site_ref_id                   site_number
      --    , subject_name                  subject_name
      -- , MAX(standard_subject_status)  standard_subject_status
      -- , MAX(subject_status)           subject_status
      , MAX(dm_director)              dm_director
      , MAX(dm_manager)               dm_manager
      , MAX(dm_study_flag)            dm_study_flag
      , MAX(lcd_manager)              lcd_manager
      , MAX(lcd_region)               lcd_region
      , MAX(project_Lead)             project_Lead
      , MAX(protocol_id)              protocol_id
      , MAX(protocol_title)           protocol_title
      from
      prc_presentation.dm_global_details
    where
    site_ref_id is NOT NULL and subject_name is NOT NULL

    group by project_id) DMT
    ON      DMT.project_code = all_sites.project_code
    ) -- end of All_PRC query
    ---------------------
    -- MAIN SELECT QUERY
      ---------------------
    SELECT   distinct
      cs.project_code
    , sf.bu_code
    , sf.owning_business_unit
    , sf.sponsor_name
    , sf.study_phase
    , pvs.study_status
    , sf.therapeutic_area
    , db.direct_budget
    , ap.site_number
    , ap.country_code3
    , ap.country_name
    , ap.region
    , ap.site_status
    , ap.sites_budgeted
    , ap.standard_subject_status
    , ap.subject_name
    , ap.subject_status
    , sf.study_name
    , sf.int_project_id int_study_id
    , ap.dm_director
    , ap.dm_manager
    , ap.dm_study_flag
    , ap.in_enrollment
    , sf.indication
    , ap.lcd_manager
    , ap.lcd_region
    , ap.milestone_actual_date
    , ap.recent_milestone_name
    , ap.pd_name
    , ap.pl_assessment
    , ap.pl_assessment_comments
    , ap.pl_assessment_date
    , ap.project_Lead
    , sf.protocol_id
    , sf.protocol_title
    , ap.last_modified_date
    , sf.sponsor_parent_company_name        -- NEW COLUMN
    , FL.SSU_PDL_Name                       -- NEW COLUMN
    , FL.Site_ID_Name                       -- NEW COLUMN
    , FL.LCRA_Name                          -- NEW COLUMN
      FROM
    Clinical_Study_All cs
      left join pv_status pvs         on (pvs.project_code) = (cs.project_code1)
    left join direct_budget db      on (db.project_code)  = (cs.project_code1)
    left join SF                    on (sf.project_code)  = (cs.project_code1)
    left join ALL_PRC AP            on (ap.project_code)  = (cs.project_code1)
    left join Functional_Lead FL    on (fl.project_code)  = (cs.project_code1)
    )  FINAL_QUERY;
  */


}