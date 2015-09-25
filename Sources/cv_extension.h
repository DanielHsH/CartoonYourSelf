/* Coded By    : Daniel Herman Shmulyan
   Version     : 2012_12
   Description : Auxillary computer vision methods and gateway to openCV. 
   Compatibility: openCv 2.4.3 and later
*/
/****************************************************************************/

// Includes
#include "opencv2\opencv.hpp"	// Resides at: \opencv\build\include\

// #include <windows.h>
#include <math.h>
#include <string> 
#include <iostream>
#include <fstream>

// Directives to linker to include openCV lib files.
#ifdef _WIN64
	// 64 bits
	#ifndef STATIC_LIBRARY_LINK
		// Linking against DLL. For each 'lib' file that appears below, final EXE will need a DLL.
		#ifdef _DEBUG
			// Core of openCV
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_core243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_ts243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_highgui243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_imgproc243d.lib") 
			//#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\zlibd.lib") 

			// Calibration and image matching
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_flann243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_features2d243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_calib3d243d.lib")

			// Object detection
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_objdetect243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Debug\\opencv_contrib243d.lib")

			// Other libs that might be needed
			/*#pragma comment(lib, "opencv_gpu220d.lib") 
			#pragma comment(lib, "opencv_video220d.lib") 
			#pragma comment(lib, "opencv_legacy220d.lib") 

			#pragma comment(lib, "opencv_ml220d.lib") 
			#pragma comment(lib, "opencv_ffmpeg220d.lib") 
			#pragma comment(lib, "opencv_contrib220d.lib") */
		#else
			//Same as above but release version
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_core243.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_ts243.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_highgui243.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_imgproc243.lib") 
			//#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\zlib.lib") 

			// Calibration and image matching
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_flann243.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_features2d243.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_calib3d243.lib")

			// Object detection
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_objdetect243.lib") 
			#pragma comment(lib, "P:\\opencv\\build\\lib\\Release\\opencv_contrib243.lib")
		#endif
	#else
		// Static linking. No DLL's would be required but EXE file will be bigger. Not recommended! Also linking in debug mode might produce many warnings since *.pdb are not always present with the lib files
   
		// Core of openCV. Must be compiled as lib and not as dll's
		#pragma comment(lib, "opencv_core.lib") 
		#pragma comment(lib, "opencv_highgui.lib") 
		#pragma comment(lib, "opencv_imgproc.lib") 

		// Calibration and image matching. Must be compiled as lib and not as dll's
		#pragma comment(lib, "opencv_flann.lib") 
		#pragma comment(lib, "opencv_features2d.lib") 
		#pragma comment(lib, "opencv_calib3d.lib") 

		// Image I/O auxillary libraries. Must be compiled as lib and not as dll's
		#pragma comment(lib, "libtiff.lib") 
		#pragma comment(lib, "libpng.lib")
		#pragma comment(lib, "zlib.lib")
		#pragma comment(lib, "libjasper.lib")
		#pragma comment(lib, "libjpeg.lib")

		// OpenCV linear algebra methods. Must be compiled as lib and not as dll's
		#pragma comment(lib, "opencv_lapack.lib")

		// Auxillary libs, found in visual studio microsoft sdk
		#pragma comment(lib, "vfw32.lib")
		#pragma comment(lib, "comctl32.lib" )
		//#pragma comment(lib, "window_w32.lib")  // Not needed
	#endif
#else
	// 32 bits
	#ifndef STATIC_LIBRARY_LINK
		// Linking against DLL. For each 'lib' file that appears below, final EXE will need a DLL.
		#ifdef _DEBUG			
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_core243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_ts243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_highgui243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_imgproc243d.lib") 
		  //#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\zlibd.lib") 

			// Calibration and image matching
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_flann243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_features2d243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_calib3d243d.lib")

			// Object detection
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_objdetect243d.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Debug\\opencv_contrib243d.lib")

		#else
			//Same as above but release version
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_core243.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_ts243.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_highgui243.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_imgproc243.lib") 
		  //#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\zlib.lib") 

			// Calibration and image matching
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_flann243.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_features2d243.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_calib3d243.lib")

			// Object detection
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_objdetect243.lib") 
			#pragma comment(lib, "P:\\opencv\\build32\\lib\\Release\\opencv_contrib243.lib")
		#endif
	#endif
#endif
#ifndef CVEXTENSION_09FJ17FB3N59SGFOL6H7H7KJD3487_IINC
#define CVEXTENSION_09FJ17FB3N59SGFOL6H7H7KJD3487_IINC

using namespace std;
using namespace cv;

/******************************** Defines ***********************************/
#define DEFAULT_PATH_LENGTH  (512)     // String length of file path

// Key Strokes
#define CVEXT_NO_KEY   (-1)
#define CVEXT_QUIT     (27)
#define CVEXT_NEXT     'n':case'N'
#define CVEXT_PAUSE    'p':case'P'
#define CVEXT_MOVE_WIN 'm':case'M'
#define CVEXT_SBS_MODE (32)


/******************************** Defines ***********************************/
// define image iterators. Use precompiler define for realtime considerations

# define cvExt1U_ImageIterator IplImageIterator<bool>
# define cvExt8U_ImageIterator IplImageIterator<unsigned char>
# define cvExt8U unsigned char
# define cvExt16UImageIterator IplImageIterator<unsigned short>
# define cvExt32FImageIterator IplImageIterator<float>
# define cvExt32F float

// Default iterator
# define cvExtImageIterator cvExt8U_ImageIterator

/***************************** cv::Mat defines ******************************/
#define NULL_MATRIX  Mat()

/****************************************************************************/
// Class Image iterator. Allows easy access to image pixels with iterator
// For example, for thresholding image use:
// cvExtImageIterator it(image);
// while (!it) {
//    *it = (*it < 30)?0:255;
//    ++it;
// }
// Image iterator runs relatively slow, unless setting compiler to optimize for speed.
// When doing so - image iterator becomes as fast as pointers manipulation

template <class PEL>
class IplImageIterator {

	  private:
			// -----------------------------------
			// Reinitialize on new image (used in constructor) 
			void initialize(IplImage* image, int startX= 0, int startY= 0, 
				                             int dX    = 0, int dY    = 0);
	  public:
			
			// -----------------------------------
			// Constructor. 
		    // Get ready to iterate from (startX,startY) up to (startX+dX,startY+dY)
		    // Coordinates are relative to cv ROI for openCV compatibility.
			IplImageIterator(IplImage* image, int startX= 0, int startY= 0, 
				                              int dX    = 0, int dY    = 0);

	  public:

			// -----------------------------------
			// Access pixel

			PEL& operator*(){
				return data[i]; 
			}

			const PEL operator*() const{
				return data[i]; 
			}


			// Get pointer to current pixel
			PEL* operator&() const;

			// -----------------------------------
			// Get current pixel coordinates

			int col() const;
			int row() const;

			// -----------------------------------
			// Access pixel neighbour
			const PEL neighbor(int dx, int dy) const;

			// -----------------------------------
			// Advance to next pixel or next color component 
			IplImageIterator& operator++();

			// Advance to next pixel or next color component, but store copy before ++
			const IplImageIterator operator++(int);

			// Jump few pixels (advanced step must be less then image width).
			// For example, use this method when you want to proccess only even pixels 
			// in each line. Note when end of line is reached iterator goes to beggining of 
			// new line disregarding size of the step.
			IplImageIterator& operator+=(int s);

			// -----------------------------------
			// Check if iterator has more data to proccess
			bool operator!() const;

			// ---------------------------------------
			// Inner image variables

	  private:
			PEL* data;			// Image information. Pointer to current row.
			int i, j;			// (i,j) coordinates of current pixel

			int start_i;		// starting column
			int step;			// number of bytes in image row
			int nRows, nCols;	// Size of the area to iterate over
			int nChannels;		// Number of channels in the image 
}; 


/****************************************************************************/
/***************************** Image Operations *****************************/
/****************************************************************************/

/****************************************************************************/
// Morphological noise removal of objects and holes of size not greater then
// minObjSize. Applies opening and then closing

void cvExtMorphologicalNoiseRemove( IplImage* img, int minObjSize, 
						            int cvShape = CV_SHAPE_RECT);

/****************************************************************************/
// Morphological skeleton transform. Just a sketch
void cvExtSkeletonTransform(IplImage* srcImg, IplImage* dstImg);

/****************************************************************************/
// Invert black/white colors of binary image {0,255}
void cvExtInvertBinaryImage(IplImage* img);

/****************************************************************************/
// Convert mask with values in {0,1} to binary image with values {0,255}
// Used for debugging
void cvExtConvertMask2Binary(IplImage* img);

/****************************************************************************/
// Strech intensity range of the image to [0..maxVal]
void cvExtIntensStrech(IplImage* img, int maxVal = 1);
void cvExtIntensStrech(Mat& img     , int maxVal = 255);

/****************************************************************************/
// Crops intensity to range [min..max]. 
// For each pixel, if (pix<min) it becomes min. Same for max
void cvExtIntensCrop(IplImage* img, int minVal = 0, int maxVal = 1);

/****************************************************************************/
// Convert hue value to RGB scalar
CvScalar cvExtHue2rgb(float hue);

/****************************************************************************/
// If Null then does nothing. Else acts like cvReleaseImage() and sets the 
// input image to NULL
void cvExtReleaseImageCheckNULL(IplImage** image);

/****************************************************************************/
// Show image on full screen
void imshowFullScreen( const string& winname, const Mat& mat );

/****************************************************************************/
// Show message box to user
void cvShowMessageBox(const char* caption, const char* message);

/****************************************************************************/
/*************************** Rectangle Operations ***************************/
/****************************************************************************/

/****************************************************************************/
// Returns 0  if two input rectangles are equal
// Returns +1 if jRect is inside iRect
// Returns -1 if iRect is inside jRect
int cvExtIsRectInside(CvRect iRect, CvRect jRect);

/****************************************************************************/
// Returns 0  if small intersection or not at all
// Returns 1  if large intersection
int cvExtIsRectLargeIntersection(CvRect iRect, CvRect jRect);
	
/****************************************************************************/
// Returns the angle of vector from center of iRect to center of jRect
// Returns also the magnitude of the vector in last parameter if it is not null
double cvExtAngBetweenRectCenters(CvRect iRect, CvRect jRect, double* vecMagnitude = NULL);

/****************************************************************************/
// Add two 2d vectors. Calculates 'vec' + 'd' into 'vec' arguments.
// Angles are given in degrees
void cvExtAddVector(double* vecMag, double* vecAng, double dMag, double dAng);

/****************************************************************************/
// Returns absolute difference between two angles (in degrees)
double cvExtAngAbsDiff(double ang1, double ang2 );

/****************************************************************************/
// Finds largest inscribed rectangle (of image proportions) inside mask 
// Mask is an arbitrary quadrilateral white shape on black background.
// Returns: (x,y) uper left point corner and (lx,ly) which are width & height.
// Uses LargestInscribedSquare()
void LargestInscribedImage(const Mat& mask, Size searchImSize, int& x, int& y, int& lx, int& ly );

/****************************************************************************/
// Input mask (white pixels are the enclosing area, blacks are background)
// Mask is char (8U) Mat with white > 0, black is = 0. 
// Returns: square location (x,y)-> (x+l,y+l)
// If mask covers convex area then faster algorithm will be launged.
void LargestInscribedSquare(const Mat& mask, int *x, int*y, int *l, bool isMaskConvex = false);

/****************************************************************************/
// Finds the corners of a given mask. Returns them in clockwise order in corners vector.
// Mask is an arbitrary quadrilateral white shape on black background.
// Best value for distThr is ~ 0.8 of mask smalles side.
// Returns the area of the detected quadrilateral
double findCornersOfMask(Mat& maskImg, int numCorners, vector<Point2f>& corners, double distThr);

/****************************************************************************/
// Finds the area of 2D convex hull (ordered vector of points).
double cvExtConvexHullArea(vector<Point2f>& ch);

/****************************************************************************/
/****************************** Image matching ******************************/
/****************************************************************************/

/****************************************************************************/
// Uses built in matching method for cross check matching. Matches src->dst, dst->src
// and findes bi-directional matches.
void cvExtCrossCheckMatching( DescriptorMatcher* matcher, const Mat& descriptors1, const Mat& descriptors2,
                              vector<DMatch>& matches, int knn=1 );

/****************************************************************************/
/*************************** Drawing on images ******************************/
/****************************************************************************/

/****************************************************************************/
// Draw arrow on the image (img) in a given location (start) in direction (angDeg),
// with length (length). Color and thikcness are also parameters.
void cvExtDrawArrow(IplImage *img, CvPoint start, double angDeg, double length,
					               CvScalar color, int thickness = 1);

/****************************************************************************/
// Draw Object rectangle with an arrow from his center pointing in defined 
// direction. Used for drawing global motion direction in swordfish debug mode.
void cvExtDrawObj(IplImage *img, CvRect objRect, double angDeg, double length,
				  CvScalar color);

/****************************************************************************/
// Visualizes a histogram as a bar graph into an output image.
// Uses different Hue for each bar.
void cvExtVisualizeHueHistogram(const CvHistogram *hist, int hdims, IplImage *img);

/****************************************************************************/
// Draw cross of size d in pixels on a specific place in the image
#define drawCrossOnImage(img, center, color, d )    \
        cvLine( img, cvPoint( center.x - d, center.y - d ),                \
                     cvPoint( center.x + d, center.y + d ), color, 1, CV_AA, 0); \
        cvLine( img, cvPoint( center.x + d, center.y - d ),                \
                     cvPoint( center.x - d, center.y + d ), color, 1, CV_AA, 0 )

/****************************************************************************/
// Convert angle to pixel at this angle around the center of the image,
// at distance of height/3
#define calcPointAroundImageCenter(img,angle)  \
        cvPoint( cvRound(img->width/2  + img->height/3*cos(angle)),  \
                 cvRound(img->height/2 - img->height/3*sin(angle))) 

/****************************************************************************/
/************************** Draw Your self **********************************/
/****************************************************************************/

// Calculates Pow 2 of absolute value of edges using soble edge detector. 
// All images must be preallocated. 
// auxImage and srcImages are changed in the function (not const)
void cvExtCalcEdgesPow2( IplImage* edgeImage, IplImage* srcImage, IplImage* auxImage);

/****************************************************************************/

// Reduces image brightness to nBit representation (2^nBits levels).
// Works on gray level IPL_DEPTH_8U images
void cvExtReduceTonBits(IplImage* srcImage, unsigned int nBits);

/****************************************************************************/

// Reduces number of levels in the image to nLeves. If 'smartLEvels' flag is set then
// this methods uses Kmeans to choose the best brightness levels.  
// Otherwise uniform levels are taken.
// If nLevels is illegal, input image is not changed
// input image must be of depth IPL_DEPTH_32F (float variable).
// Output is in range [0..255].
void cvExtReduceLevels(IplImage* srcImage, unsigned int nLevels = 0, bool smartLevels = true);

/****************************************************************************/

// Convert image to pencil sketch
// Parameters: 
//        intensNormal is in [0..1]   - normalizes variations in image intensity
//        looseDitails is 0 or grater - Defines how much detailes will be in the image
//                                      0 - Best quality. High value means less ditails
//        pencilLines  is 0 or grater - Defines the size of bold black pencil lines
//        colorNormal  is 0 or grater - Gamma correction to color depth (saturation)
//        colorDepth   is in [1..8]   - Number of bits to represent color
//        fast         is 0 or 1      - Defines type of algorithm
void cvExtPhoto2Pencil(const IplImage* srcImage, IplImage* dstImage, 
					   double intensNormal = 0.9,
					   int    looseDitails = 2,
					   int    pencilLines  = 8,
					   int    colorNormal  =15,
					   int    colorBits    = 3,
					   int    fast         = 8);

/****************************************************************************/
// I/O
/****************************************************************************/

// Check if output directory exists. If not, create it. Returns true if OK.
bool createDirIfNotExists(char *DirPath);

/****************************************************************************/
bool isFileExists(char *filePath);

/****************************************************************************/
/********************* Media Frame Grabber & Writer *************************/
/****************************************************************************/

// SINGLE_IMAGE  - read/write of single image to/from storage device.                   Like "c:\a.jpg"
// ONLINE_CAMERA - read only video stream from camera.
// ONLINE_WINDOW - write only video stream or single image to a window.                 Like "Figure1@win"
// OFFLINE_VIDEO - read/write video file to/from storage device.                        Like "c:\a.mov"
// FRAMES_TEXT_LIST - read/write sequence of images whos path's is written in txt file. Like "c:\a.txt"
// DEPLETED      - no more images in the reading source
// UNKNOWN       - Error in detecting input/output media type
// MAT_PTR_DEBUG - read from pointer to Mat variable. Value of Mat is changing outside reader. Used for debug
typedef enum { SINGLE_IMAGE = -1, ONLINE_CAMERA = 0, ONLINE_WINDOW, OFFLINE_VIDEO = 100, FRAMES_TEXT_LIST = 101, DEPLETED, 
               UNKNOWN = -200, MAT_PTR_DEBUG = -300 } cvExtMediaType;

#define CVEXT_WRONG_FPS   (0)  
#define CVEXT_WRONG_CODEC (-178)  

// Detect which type of media is stored in the given file path
cvExtMediaType cvExtDetectMediaType(const string path);

/****************************************************************************/
// Grabs frames from video files, images and list of images
class cvExtFrameGrabber{

      public:
		    
		    cvExtFrameGrabber(int camIndex = 0);    // Constructor for online grabbing from camera
            cvExtFrameGrabber(string path);         // Constructor for offline grabbing from media file
            cvExtFrameGrabber(const Mat*sourceMat); // Constructor for online grabbing from matrix (which can be changed from outside)
			                                        // Mostly used for debug

            ~cvExtFrameGrabber();			        // Destructor

			// Returns current frame. User may change the image but also responsible for releasing it.
			Mat       grabFrame(void);
			IplImage* grabFrameIpl(void);

			// Returns current frame. User must NOT release or change the image. Returned image will dissapear
			// when next frame arrives or this object is destroyied.
			const Mat      watchFrame(void);
			const IplImage watchFrameIpl(void);

			// Change video flipping flag. If flag is set then video will be flipped
			void setVerticalFlip(bool flag);
			void setHorizontalFlip(bool flag);

			// Returns the number of current frame
			int getCurFrameNumber(void);

			// Return true if current video source is online (real time)
			bool isRealTimeGrabbing(void);

			// Return true if current source is video and not sequence of images.
			bool isVideoStreamGrabbing(void);

			double getFPS(void);                        // Return media frames per second ratio
			int    getMediaCodec(void);                 // Return media decompressing codec
			void   getMediaSize(int*width, int*height); // Get image size. Works only for cameras
			bool   setMediaSize(int width, int height); // Set image size. Works only for cameras

			// Activate undistortion on the media. Distortion parameters are stored in file in camera.xml:
			// "intrCamMat" - Intrinsic matrix, "distortion" - 4 distortion parameters, and "avgReprojectionErr" value
			// Returns true if undistortion could be activated
			bool activateUndistort(const string filePath);
     private:
	
			// Frame grabber flags
			int verticalFlipFlag, horizontalFlipFlag;
            int frameNumber;
			int isUndistortActivated;			// Undistortion parameters
			
		    cvExtMediaType framesSource;		// Source of frames.		    
		    string videoFilePath;				// Path to video file that will be read (only offline grabbing)			
			const Mat*  sourceMat;				// Used for online grabbing from matrix (which can be changed from outside).
			FILE* textFile;
			char imageName[DEFAULT_PATH_LENGTH];// Text file with images list (only offline grabbing)
			
			//IplImage*  curFrame;
			Mat  curFrame;						// Pointer to current frame data
			Mat IntrinsicMat, DistCoef;
			
		    VideoCapture capture;				// CV capture structure
			int    codec;						// Codec of the video
			double FPS;							// Frames per second ratio
};

/****************************************************************************/
// Writes frames to video files, images and list of images
class cvExtFrameWriter{

      public:
		    
			cvExtFrameWriter(string path, double fps = CVEXT_WRONG_FPS, int codec = CVEXT_WRONG_CODEC, bool isColored = true);
            ~cvExtFrameWriter();			  // Destructor

			// Writes current frame.
			void writeFrame(const Mat&      frame);
			void writeFrame(const IplImage* frame);   // Backword compatibility

			// Return true if current source is video and not sequence of images.
			bool isVideoStreamWriting(void);

      private:
		   
		    cvExtMediaType framesDest;  // Destination of frames.

		    // Path to video file that will be written 
		    string videoFilePath;	 
		    size_t dirEnd;  // Separate path directory from file name

			// Text file with images list 
			FILE* textFile;
			string imageName;

			// CV video writer structure
			VideoWriter writer;
			double vFPS;
			bool   vIsColored; 
			int    vCodec;

			// Frame writing flags
			bool verticalFlipFlag;
			bool horizontalFlipFlag;
		
            int frameNumber;   // Frame number
};

/****************************************************************************/
/***************************** cv:Mat Operations ****************************/
/****************************************************************************/

// Apply transformation T to column vectors v. V is not in homogenous coordinates
void cvExtTransform(const Mat& T, Mat& v);

/****************************************************************************/
// Check that 3x3 homography T is a valid transformation.
// Checks inner values of T, and that T transforms all pixels in image of srcSize inside dstSize
bool cvExtCheckTransformValid(const Mat& T, Size srcSize = Size(0,0), Size dstSize = Size(0,0));

/****************************************************************************/
// Build vector (x,y) for each image corner and center. Store the reult is 2x5 matrix.
void cvExtBuildFootPrintColsVector(const Size s, Mat& footPrint);

// Build vector (x,y) for each image corner. Store the reult is 4x2 matrix.
void cvExtBuildCornersRowsVector(const Size s, Mat& cornersVec);

/****************************************************************************/
// Calculate normalized location of the projector that created the given footprint.
// Range: -Inf..Inf  Large X means right, Large Y means down. (0,0) is center
void cvExtFootPrintSourceLocation(Mat& footPrint, float& Xindex, float& Yindex);

/****************************************************************************/
// Returns 3x3 perspective transformation for the corresponding 4 point pairs, 
// stored as row vectors in 4x2 matrix.
Mat cvExtgetPerspectiveTransform(const Mat& SrcRowVec, const Mat& DstRowVec);

/****************************************************************************/
// Print Matrix. if isInt is true, prints round numbers. Otherwise prints as fractions.

template <typename T>  
void printMatTemplate(const Mat& M, bool isInt = true){
	if (M.empty()){
	   printf("Empty Matrix\n");
	   return;
	}
	if ((M.elemSize()/M.channels()) != sizeof(T)){
	   printf("Wrong matrix type. Cannot print\n");
	   return;
	}
	int cols = M.cols;
	int rows = M.rows;
	int chan = M.channels();

	char printf_fmt[20];
	if (isInt)
	   sprintf_s(printf_fmt,"%%d,");
	else
	   sprintf_s(printf_fmt,"%%0.5g,");
	
	if (chan > 1){
		// Print multi channel array
		for (int i = 0; i < rows; i++){
			for (int j = 0; j < cols; j++){	        
				printf("(");
				const T* Pix = &M.at<T>(i,j);
				for (int c = 0; c < chan; c++){
				   printf(printf_fmt,Pix[c]);
				}
				printf(")");
			}
			printf("\n");
		}
		printf("-----------------\n");			
	}
	else {
		// Single channel
		for (int i = 0; i < rows; i++){
			const T* Mi = M.ptr<T>(i);
			for (int j = 0; j < cols; j++){
			   printf(printf_fmt,Mi[j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

/****************************************************************************/
int inline CV_BUILD_MATRIX_TYPE(int elementSize, int nChannels){
	// Determine type of the matrix 
	switch (elementSize){
	case sizeof(char):
		 return CV_8UC(nChannels);
  		 break;
	case (2*sizeof(char)):
		 return CV_16UC(nChannels);
  		 break;
	case sizeof(float):
		 return CV_32FC(nChannels);
  		 break;
	case sizeof(double):
		 return CV_64FC(nChannels);
  		 break;
	}
	return -1;
}

/****************************************************************************/

// Serialize matrix to/from char buffer or to/from file.
// char* buf is buffer for serialization or path to serialization file. 
// When: writing to buffer - it is allocated and must be freed by user.
// Returns the number of bytes that were serialized
#define CVEXT_SERIALIZE_READ    (0)    // default
#define CVEXT_SERIALIZE_WRITE   (1)
#define CVEXT_SERIALIZE_CHARBUF (0)    // Default
#define CVEXT_SERIALIZE_FILE    (2)
#define CVEXT_SERIALIZE_MAT_FROM_CHARBUF   ( CVEXT_SERIALIZE_READ|CVEXT_SERIALIZE_CHARBUF)
#define CVEXT_SERIALIZE_MAT_TO_CHARBUF     (CVEXT_SERIALIZE_WRITE|CVEXT_SERIALIZE_CHARBUF)
int serializeMat(Mat& M, char* &buf, char serializeFlags);

// Save/Read matrix to binary file. Wrappe the serializeMat() method
int saveMat(char* filename, Mat& M);
int readMat(char* filename, Mat& M);


/****************************************************************************/
// Serialize char buffer to/from file.
// serializeFlags are CVEXT_SERIALIZE_READ for reading from file to buf or CVEXT_SERIALIZE_WRITE.
// When: reading from file to buffer - it is allocated and must be freed by user.
// When: writing from buf to file buSize must be specified.
// Returns the number of bytes that were serialized.
int serializeCharBuf(char* &buf, char* filePath, char serializeFlags, int bufSize = 0);

/****************************************************************************/
// Embedd char buffer in cv matrix or extract buffer from it. 
// Lets you pass any type of data using Mat container
// Returns the number of bytes that were embedded or extracted
int embeddCharDataInMat(Mat& M, char* buf, int bufSize);
int extractCharDataFromMat(Mat& M, char* &buf, int &bufSize);

/****************************************************************************/
void printMat(const Mat& M);
void deb(const Mat& M);
/****************************************************************************/
// Converts 4x2 double matrix into array of 4 Point2f. 
// 'dst' must be preallocated.
void cvtCornersMat_Point2fArray(const Mat& M, Point2f *dst);

	
/****************************************************************************/
/*********************************** Other **********************************/
/****************************************************************************/

/****************************************************************************/
// Simple Min/Max function 
#define CVEXT_MAX(a,b)  (a>b)?a:b
#define CVEXT_MIN(a,b)  (a<b)?a:b

/****************************************************************************/
// integer to boolean conversion
#define CVEXT_INT_2_BOOL(a)  ((a>0)?true:false)

/****************************************************************************/
// String copy with protection against buffer overflow (by truncating suffix)
inline void safeStrCpy_(char* dst, const char* src, int dstSize){
	int i = 0;
	
	// Verify that source is not NULL
	if (src)
	   for (; (i < dstSize-1)&&(src[i]); *(dst++) = src[i++]);		
	*dst = '\0';
}
#define safeStrBufCpy(dst,src)  safeStrCpy_(dst,src,sizeof(dst)) 

/****************************************************************************/
// Verify that variable has a legal character value [0..255]
#define verifyCharRange(v)  \
		if (v<0) v = 0; if (v>255) v = 255;

/****************************************************************************/

#endif // #define CVEXTENSION_09FJ17FB3N59SGFOL6H7H7KJD3487_IINC

/****************************************************************************/
// EOF.
