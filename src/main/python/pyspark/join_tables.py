from __future__ import print_function

from pyspark import SparkContext
from pyspark.sql import SparkSession

# sample.data.txt
# date,region_code,players
# 2005-10-30,101,sachin:rahul:kallis
# 2005-10-25,101,sachin:abd::
# 2006-01-15,102,younis:yuvraj:dhoni
# 2006-01-18,102,yuvraj:sachin
# 2006-01-15,102,younis:yuvraj:dhoni
# 2007-05-20,101,sehwag:sachin:ponting::
# 2007-05-25,101,hayden:yuvraj:ponting
# 2007-05-30,101,clarke:sehwag::ponting
# 2008-02-20,103,jaysurya:sangkara:ganguly::
# 2008-02-25,103,sachin:yuvraj:attpattu
# 2008-02-30,103,sachin:sehwag:tharanga

# sample.lookup.txt
# region_code,country
# 101,india
# 102,pakistan
# 103,srilanka
# 104,australia
# 105,south_africa


sc = SparkContext()
spark = SparkSession.builder.appName("Joining_2_tables").getOrCreate()
counter = sc.accumulator(0)


def myFunc(a):
    list1 = a
    global counter
    myList = []
    myList1 = []
    for l in list1:
        for m in l:
            myList1.append(m)
            if (m != ''):
                myList.append(m)

    if ('' in myList1):
        counter += 1
    return (max(myList, key=myList.count))


def myFunc2(a):
    in1 = a
    myList = []
    for l in in1[0]:
        myList.append(l)
    myList.append(in1[1])
    t = tuple(myList)
    return t


input1 = sc.textFile("file:///home/hduser/sample.data.txt")
t0 = input1.first()
t1 = input1.filter(lambda x: x != t0)
t2 = t1.map(lambda x: str(x).split(",")).map(lambda x: (x[0].split("-")[0], x[1], x[2])).map(
    lambda x: (x[0] + '-' + x[1], x[2])).mapValues(lambda x: x.split(":"))
t3 = t2.groupByKey().mapValues(lambda x: list(x))
t4 = t3.mapValues(lambda x: myFunc(x))
t5 = t4.map(lambda x: (x[0].split("-"), x[1])).map(lambda x: myFunc2(x))
t6 = t5.toDF()
t7 = t6.createTempView("table1")

input2 = sc.textFile("file:///home/hduser/sample.lookup.txt")
h1 = input2.first()
h2 = input2.filter(lambda x: x != h1)
h3 = h2.map(lambda x: str(x).split(","))
h4 = h3.toDF()
h5 = h4.createTempView("table2")
th = spark.sql(
    "select table2._2 as country,table1._1 as year,table1._3 as player_scored_most_century from table1 left join table2 on table1._2=table2._1")
th.show()
print("no-of-bad-records = ", counter.value)
