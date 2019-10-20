package utils

import com.typesafe.scalalogging.LazyLogging
import play.api.libs.json.{JsValue, Json}
import scalaj.http.{Http, HttpOptions, HttpRequest, HttpResponse}

import scala.collection.immutable.List
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

/* http utility functions using scalaj*/

object HttpUtils extends Enumeration with LazyLogging {

  type requestType = Value
  lazy val frameHeaders: () => Map[String, String] = () => {
    val authDetails: JsValue = Json.obj()
    Json.fromJson[Map[String, String]](authDetails).getOrElse(Map.empty[String, String]) + ("Content-Type" -> "application/json")
  }
  val GET, POST, PUT = Value

  def createSolrQuery(search: Option[List[String]] = Some(List(""""*:*"""")), filter: Option[List[String]] = None,
                      id: Option[String] = None, from: Option[Int] = None,
                      size: Option[Int] = Some(1000), paginate: Boolean = false,
                      cursorMark: Option[String] = None): String = {

    implicit val emptyStr: String = ""

    implicit def optionToString[T: ClassTag](option: Option[T])(implicit emptyString: String): String = {
      option match {
        case Some(value) =>
          value match {
            case x: List[_] => x.mkString(",")
            case _ => value.toString
          }
        case None => emptyString
      }
    }

    implicit def optionToInt(option: Option[Int]): Int = {
      option match {
        case Some(value) => value
        case None => 0
      }
    }

    val searchValue: String = search
    val filterValue: String = filter
    val idValue: String = id
    val fromValue: Int = from
    val sizeValue: Int = size
    val cursorMarkValue: String = cursorMark

    s"""{
       |  "sq": {
       |    "q": [ $searchValue ], "fq": [ $filterValue ]
       |  },
       |  "ids": [ $idValue ],
       |  "from": $fromValue,
       |  "size": $sizeValue,
       |  "paginate": $paginate,
       |  "cursor_mark": "$cursorMarkValue"
       |}""".stripMargin
  }

  @scala.annotation.tailrec
  def retry(request: HttpRequest, retriesLeft: Int = 2): HttpResponse[String] = {
    val response: Try[HttpResponse[String]] = Try(request.asString)
    if (retriesLeft <= 0) {
      response.get
    } else {
      response match {
        case Success(successResponse) if (200 until 300).contains(successResponse.code) => successResponse
        case Success(nonRetryable) if (400 until 500).contains(nonRetryable.code) =>
          logger.error(s"Received response ${nonRetryable.code} for url ${request.url}")
          nonRetryable
        case Success(retryResponse) if (500 until 600).contains(retryResponse.code) =>
          logger.warn(s"Received response ${retryResponse.code} for url ${request.url}. Retrying again")
          Thread.sleep(1000)
          retry(request, retriesLeft - 1)
        case Failure(exception) =>
          logger.error(s"exception for $request - ${exception.printStackTrace()}")
          Thread.sleep(1000)
          retry(request, retriesLeft - 1)
      }
    }
  }

  def getResponse(api: String, requestType: requestType = GET, body: Option[String] = None, retries: Int = 2)
  : HttpResponse[String] = {
    val request: HttpRequest = requestType match {
      case GET => Http(api).headers(frameHeaders())
        .option(HttpOptions.connTimeout(10000)).option(HttpOptions.readTimeout(60000))
      case POST => body match {
        case Some(postBody) =>
          Http(api).postData(postBody).headers(frameHeaders())
            .option(HttpOptions.connTimeout(10000)).option(HttpOptions.readTimeout(60000))
        case None => throw new Exception(s"empty body in post for $api")
      }
      case PUT => body match {
        case Some(putBody) =>
          Http(api).put(putBody).headers(frameHeaders())
            .option(HttpOptions.connTimeout(10000)).option(HttpOptions.readTimeout(60000))
        case None => throw new Exception(s"empty body in post for $api")
      }
    }
    retry(request, retries)
  }

}
