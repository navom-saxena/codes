package akka

import akka.actors.AkkaInMemDbClient
import org.scalatest.{FunSpecLike, Matchers}

import scala.concurrent.duration._
import scala.concurrent.{Await, Future}

/* akka client test which creates a client object and asks for input and response. AkkaInMemDbClient holds reference of
*  remote AkkaInMemDbClient which runs on remoteAddress.
* */

class AkkaInMemDbClientTest extends FunSpecLike with Matchers {

  val client = new AkkaInMemDbClient("127.0.0.1:59885")

  describe("AkkaInMemDbClient") {

    client.set("123", "123-value")
    val futureResponse: Future[Any] = client.get("123")
    val response: String = Await.result(futureResponse.mapTo[String], 10 seconds)
    response should equal("123-value")

  }

}
