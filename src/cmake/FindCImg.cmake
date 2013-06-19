FIND_PATH(
	CIMG_INCLUDE_DIR CImg.h
          /usr/lib/CImg
          /usr/local/lib/CImg
          /opt/local/lib/CImg
	"${CMAKE_CURRENT_SOURCE_DIR}/CImg"
         )

SET(CIMG_FOUND FALSE)
IF(CIMG_INCLUDE_DIR)
  SET(CIMG_FOUND TRUE)
  INCLUDE_DIRECTORIES(${CIMG_INCLUDE_DIR})
ELSE(CIMG_INCLUDE_DIR)
  MESSAGE(FATAL ERROR "CImg.h not found - Please install CImg")
ENDIF(CIMG_INCLUDE_DIR)
