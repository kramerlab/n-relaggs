-- MySQL dump 10.13  Distrib 5.7.26, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: trains
-- ------------------------------------------------------
-- Server version	5.5.5-10.3.15-MariaDB-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `cars`
--

DROP TABLE IF EXISTS `cars`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `cars` (
  `id` int(11) NOT NULL,
  `train_id` int(11) DEFAULT NULL,
  `position` int(11) DEFAULT NULL,
  `shape` varchar(255) DEFAULT NULL,
  `len` varchar(255) DEFAULT NULL,
  `sides` varchar(255) DEFAULT NULL,
  `roof` varchar(255) DEFAULT NULL,
  `wheels` int(11) DEFAULT NULL,
  `load_shape` varchar(255) DEFAULT NULL,
  `load_num` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `train_id` (`train_id`),
  CONSTRAINT `cars_ibfk_1` FOREIGN KEY (`train_id`) REFERENCES `trains` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `cars`
--

LOCK TABLES `cars` WRITE;
/*!40000 ALTER TABLE `cars` DISABLE KEYS */;
INSERT INTO `cars` VALUES (1,1,1,'rectangle','short','not_double','none',2,'circle',1),(2,1,2,'rectangle','long','not_double','none',3,'hexagon',1),(3,1,3,'rectangle','short','not_double','peaked',2,'triangle',1),(4,1,4,'rectangle','long','not_double','none',2,'rectangle',3),(5,2,1,'rectangle','short','not_double','flat',2,'circle',2),(6,2,2,'bucket','short','not_double','none',2,'rectangle',1),(7,2,3,'u_shaped','short','not_double','none',2,'triangle',1),(8,3,1,'rectangle','long','not_double','flat',3,'triangle',1),(9,3,2,'hexagon','short','not_double','flat',2,'triangle',1),(10,3,3,'rectangle','short','not_double','none',2,'circle',1),(11,4,1,'rectangle','short','not_double','none',2,'rectangle',1),(12,4,2,'ellipse','short','not_double','arc',2,'diamond',1),(13,4,3,'rectangle','short','double','none',2,'triangle',1),(14,4,4,'bucket','short','not_double','none',2,'triangle',1),(15,5,1,'rectangle','short','not_double','flat',2,'circle',1),(16,5,2,'rectangle','long','not_double','flat',3,'rectangle',1),(17,5,3,'rectangle','short','double','none',2,'triangle',1),(18,6,1,'rectangle','long','not_double','jagged',3,'rectangle',1),(19,6,2,'hexagon','short','not_double','flat',2,'circle',1),(20,6,3,'rectangle','short','not_double','none',2,'triangle',1),(21,6,4,'rectangle','long','not_double','jagged',2,'rectangle',0),(22,7,1,'rectangle','long','not_double','none',2,'hexagon',1),(23,7,2,'rectangle','short','not_double','none',2,'rectangle',1),(24,7,3,'rectangle','short','not_double','flat',2,'triangle',1),(25,8,1,'rectangle','short','not_double','peaked',2,'rectangle',1),(26,8,2,'bucket','short','not_double','none',2,'rectangle',1),(27,8,3,'rectangle','long','not_double','flat',2,'circle',1),(28,8,4,'rectangle','short','not_double','none',2,'rectangle',1),(29,9,1,'rectangle','long','not_double','none',2,'rectangle',3),(30,9,2,'rectangle','short','not_double','none',2,'circle',1),(31,9,3,'rectangle','long','not_double','jagged',3,'hexagon',1),(32,9,4,'u_shaped','short','not_double','none',2,'triangle',1),(33,10,1,'bucket','short','not_double','none',2,'triangle',1),(34,10,2,'u_shaped','short','not_double','none',2,'circle',1),(35,10,3,'rectangle','short','not_double','none',2,'triangle',1),(36,10,4,'rectangle','short','not_double','none',2,'triangle',1),(37,11,1,'rectangle','short','not_double','none',2,'triangle',1),(38,11,2,'rectangle','long','not_double','flat',2,'circle',3),(39,12,1,'rectangle','long','not_double','jagged',2,'circle',0),(40,12,2,'u_shaped','short','not_double','none',2,'triangle',1),(41,12,3,'rectangle','short','double','none',2,'circle',1),(42,13,1,'u_shaped','short','not_double','none',2,'circle',1),(43,13,2,'rectangle','long','not_double','flat',3,'rectangle',1),(44,14,1,'bucket','short','not_double','none',2,'circle',1),(45,14,2,'rectangle','short','not_double','none',2,'rectangle',1),(46,14,3,'rectangle','long','not_double','jagged',3,'rectangle',1),(47,14,4,'bucket','short','not_double','none',2,'circle',1),(48,15,1,'rectangle','long','not_double','none',2,'rectangle',2),(49,15,2,'u_shaped','short','not_double','none',2,'rectangle',1),(50,16,1,'bucket','short','not_double','none',2,'rectangle',1),(51,16,2,'rectangle','long','not_double','flat',2,'triangle',3),(52,17,1,'rectangle','long','not_double','none',2,'hexagon',1),(53,17,2,'rectangle','short','not_double','none',2,'circle',1),(54,17,3,'rectangle','short','double','none',2,'circle',1),(55,17,4,'rectangle','long','not_double','none',2,'rectangle',3),(56,18,1,'u_shaped','short','not_double','none',2,'triangle',1),(57,18,2,'rectangle','long','not_double','none',3,'rectangle',3),(58,19,1,'rectangle','long','not_double','flat',3,'rectangle',3),(59,19,2,'rectangle','long','not_double','flat',2,'rectangle',3),(60,19,3,'rectangle','long','not_double','none',2,'rectangle',0),(61,19,4,'u_shaped','short','not_double','none',2,'triangle',1),(62,20,1,'rectangle','long','not_double','flat',3,'hexagon',1),(63,20,2,'u_shaped','short','not_double','none',2,'triangle',1);
/*!40000 ALTER TABLE `cars` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trains`
--

DROP TABLE IF EXISTS `trains`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `trains` (
  `id` int(11) NOT NULL,
  `direction` varchar(4) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `id` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trains`
--

LOCK TABLES `trains` WRITE;
/*!40000 ALTER TABLE `trains` DISABLE KEYS */;
INSERT INTO `trains` VALUES (1,'east'),(2,'east'),(3,'east'),(4,'east'),(5,'east'),(6,'east'),(7,'east'),(8,'east'),(9,'east'),(10,'east'),(11,'west'),(12,'west'),(13,'west'),(14,'west'),(15,'west'),(16,'west'),(17,'west'),(18,'west'),(19,'west'),(20,'west');
/*!40000 ALTER TABLE `trains` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-06-19 10:34:24
