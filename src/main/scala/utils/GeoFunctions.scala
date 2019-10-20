package utils

import com.typesafe.scalalogging.LazyLogging
import com.vividsolutions.jts.geom._
import com.vividsolutions.jts.io.WKBReader

import scala.collection.immutable
import scala.collection.immutable.List

// Geometry functions to calculate area, length and other geo information from wkb-(well-known-binary)

object GeoFunctions extends LazyLogging {

  def createPoint(latitude: Double, longitude: Double): Point = {
    val gf: GeometryFactory = new GeometryFactory()
    gf.createPoint(new Coordinate(longitude * 1e-7, latitude * 1e-7))
  }

  /**
   * Checks if the given WKB is a polygon or polyline and calulates the area and length
   *
   * @param wkb Geometry wkb
   * @return The area and length as tuple
   */
  def calculateWkbAreaLength(wkb: Option[String]): (Option[Double], Option[Double]) = {
    try {
      val nonPolygon: Geometry = new WKBReader().read(WKBReader.hexToBytes(wkb.get))
      val polygon: Geometry = convertWkbToGeometry(wkb)
      val featureArea: Double = getArea(polygon)
      val featureLength: Double = getLength(nonPolygon)
      (Some(featureArea), Some(featureLength))
    }
    catch {
      case e: Exception =>
        e.printStackTrace()
        logger.error(s"exception while calculating area/length. Fallback to (0,0)")
        (Some(0.0), Some(0.0))
    }
  }

  /** Calculates area for a given geometry
   *
   * @param geom Input Geometry object
   * @return area of input geometry
   */
  def getArea(geom: Geometry): Double = {
    val _90_degree: Double = 90.0
    val quarterEarthDiameter: Double = 1.0e7
    val degreePerMeter: Double = _90_degree / quarterEarthDiameter
    val meterPerDegree: Double = 1 / degreePerMeter

    val areaInDegree: Double = geom.getArea
    val latitude: Double = geom.getCentroid.getY
    val metric: Double = meterPerDegree * meterPerDegree * Math.cos(latitude / 180.0 * Math.PI)
    areaInDegree * metric
  }

  /** Calculates length for a given geometry
   *
   * @param geom Input Geometry object
   * @return length of input geometry
   */
  def getLength(geom: Geometry): Double = {
    val R: Double = 6378137.toDouble

    def calcNormalizedDist(fromLat: Double, fromLon: Double, toLat: Double, toLon: Double): Double = {
      val sinDeltaLat: Double = Math.sin(Math.toRadians(toLat - fromLat) / 2)
      val sinDeltaLon: Double = Math.sin(Math.toRadians(toLon - fromLon) / 2)
      sinDeltaLat * sinDeltaLat + sinDeltaLon * sinDeltaLon * Math.cos(Math.toRadians(fromLat)) * Math.cos(Math.toRadians(toLat))
    }

    def calcDist(fromLat: Double, fromLon: Double, toLat: Double, toLon: Double): Double = {
      val normedDistance: Double = calcNormalizedDist(fromLat, fromLon, toLat, toLon)
      R * 2 * Math.asin(Math.sqrt(normedDistance))
    }

    geom match {
      case lineString: LineString =>
        logger.whenDebugEnabled(s"geometry type is LineString - $lineString")
        lineString.getCoordinates.sliding(2).foldLeft(0.0) {
          case (length: Double, Array(c0, c1)) => length + calcDist(c0.y, c0.x, c1.y, c1.x)
        }
      case polygon: Polygon =>
        logger.whenDebugEnabled(s"geometry type is LineString - $polygon")
        polygon.getExteriorRing.getCoordinates.sliding(2).foldLeft(0.0) {
          case (length: Double, Array(c0, c1)) => length + calcDist(c0.y, c0.x, c1.y, c1.x)
        }
      case geometry =>
        logger.whenDebugEnabled(s"geometry type is ${geometry.getGeometryType} - $geometry")
        geometry.getLength * (Math.PI / 180) * R
    }
  }

  def convertWkbToGeometry(wkb: Option[String]): Geometry = {
    require(wkb.isDefined && wkb.nonEmpty)
    val nonPolygonGeometry: Geometry = new WKBReader().read(WKBReader.hexToBytes(wkb.get))
    lazy val coordArray: Array[Coordinate] = nonPolygonGeometry.asInstanceOf[LineString].getCoordinateSequence.toCoordinateArray
    nonPolygonGeometry match {
      case lineString: LineString if coordArray.head == coordArray.last && coordArray.length > 4 =>
        logger.whenDebugEnabled(s"geometry type is ${lineString.getGeometryType}. Geometry is closed.")
        convertGeometryToPolygon(lineString)
      case geometry: Geometry =>
        logger.whenDebugEnabled(s"geometry type is ${geometry.getGeometryType}")
        geometry
    }
  }

  /** Converts polyline to polygon if its closed geometry.
   *
   * @param geom Input geometry object
   * @return The polygon geometry
   */
  def convertGeometryToPolygon(geom: Geometry): Polygon = {
    require(geom.isInstanceOf[LineString], s"Found ${geom.toText}")
    new GeometryFactory().createPolygon(geom.asInstanceOf[LineString].getCoordinateSequence)
  }

  def getGeometriesFromGeometryCollection(geometryCollection: Geometry): immutable.Seq[Geometry] = {
    geometryCollection match {
      case geometry: GeometryCollection =>
        val numGeometries: Int = geometry.getNumGeometries
        (0 until numGeometries map geometry.getGeometryN filterNot (_.isInstanceOf[Point])).toList
      case notGeometryCollection: Geometry => List(notGeometryCollection)
    }
  }

}
