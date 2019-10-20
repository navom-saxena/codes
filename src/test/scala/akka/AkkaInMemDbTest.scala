package akka

import akka.actor.ActorSystem
import akka.actors.AkkaInMemDb
import akka.messages.Requests._
import akka.testkit.TestActorRef
import org.scalatest.{BeforeAndAfterEach, FunSpecLike, Matchers}

class AkkaInMemDbTest extends FunSpecLike with Matchers with BeforeAndAfterEach {

  implicit val system: ActorSystem = ActorSystem()

  describe("AkkaInMemDb") {
    describe("given SetRequest") {
      it("should place key/value into map") {

        val actorRef: TestActorRef[AkkaInMemDb] = TestActorRef(new AkkaInMemDb)
        //      or get actorRef through system.actorOf(Props(classOf[AkkaInMemDb]))
        println(actorRef.path)
        actorRef ! SetRequest("someKey", "someValue")
        val akkaInMemDb: AkkaInMemDb = actorRef.underlyingActor

        akkaInMemDb.map.get("someKey") should equal(Some("someValue"))

      }
    }
  }

}
