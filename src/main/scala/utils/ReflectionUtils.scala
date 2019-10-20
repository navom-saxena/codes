package utils

import scala.reflect.runtime.{universe => ru}
import scala.tools.reflect.ToolBox

object ReflectionUtils {

  val runtimeMirror: ru.Mirror = ru.runtimeMirror(getClass.getClassLoader)
  val tb: ToolBox[_] = runtimeMirror.mkToolBox()

  def compileCode(code: String): Unit = {
    val tree = tb.parse(code)
    tb.compile(tree)
  }

  def compileCodeWithType[T](code: String): T = {
    val tree = tb.parse(code)
    val compiledCode: () => Any = tb.compile(tree)
    compiledCode().asInstanceOf[T]
  }

  def runCode(code: String): Unit = {
    val tree = tb.parse(code)
    tb.eval(tree)
  }
}

