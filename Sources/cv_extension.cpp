// See Header in H file
/****************************************************************************/

// Includes
#include "cv_extension.h"
#include <windows.h>
/****************************************************************************/
/*********************** Image iterator implementation **********************/
/****************************************************************************/
// Constructor. 
template <class PEL>
IplImageIterator<PEL>::IplImageIterator(IplImage* image, int startX= 0, int startY= 0, 
	                              int dX    = 0, int dY    = 0){
	initialize(image, startX, startY, dX, dY);
}

/****************************************************************************/
// Reinitialize on new image
template <class PEL>
void IplImageIterator<PEL>::initialize(IplImage* image, int startX= 0, int startY= 0, 
	                                                    int dX    = 0, int dY    = 0){ 

	// Read image information
	data        = reinterpret_cast<PEL*>(image->imageData); 
	step        = image->widthStep / sizeof(PEL);
	nChannels   = image->nChannels;

	// Retrieve infromation from image ROI
	CvRect rect = cvGetImageROI(image);
	nRows = rect.height;
	nCols = rect.width;

	// Move (startX,startY) to be relative to ROI upper left pixel
	startX+= rect.x;
	startY+= rect.y;

	// Rows to proccess is min(nRows,startY+dY)
	if ((startY+dY)>0 && (startY+dY)<nRows){
  		nRows= startY+dY;
	}

	// Check validity:  0 <= startY < nRows
	if (startY<0 || startY>=nRows){ 
		startY=0;
	}
	j = startY;
	
	// Skip rows if j > 0.
	data+= step*j;
 
	// Initialize values
	i = startX;
	
	// Cols to proccess is min(nCols,startX+dX)
	if ((startX+dX)>0 && (startX+dX)<nCols){ 
	   nCols= startX+dX;
	}

	// Check validity:  0 <= startX < nCols			
	if (startX<0 || startX>=nCols) {
	   startX = 0;
	}
	nCols *= nChannels;
	i = start_i = startX*nChannels;
}

/****************************************************************************/
// Get pointer to current pixel
template <class PEL>
PEL* IplImageIterator<PEL>::operator&() const { 
	return data+i; 
}
/****************************************************************************/
// Get current pixel coordinates

template <class PEL>
int IplImageIterator<PEL>::col() const{ 
	return i/nChannels; 
}

template <class PEL>
int IplImageIterator<PEL>::row() const{ 
	return j; 
}
/****************************************************************************/
// Access pixel neighbour

template <class PEL>
const PEL IplImageIterator<PEL>::neighbor(int dx, int dy) const { 
	return *( data + (dy*step) + i + dx*nChannels ); 
}

/****************************************************************************/
// Advance to next pixel or next color component 
template <class PEL>
IplImageIterator<PEL>& IplImageIterator<PEL>::operator++(){
		i++; 
		// Check if end of row is reached
		if (i >= nCols){ 
			i = start_i; 
			// Go to next line
			j++; 
			data+= step; 
		}
		return *this;
}

/****************************************************************************/
// Advance to next pixel or next color component, but store copy before ++
template <class PEL>
const IplImageIterator<PEL> IplImageIterator<PEL>::operator++(int){
		IplImageIterator<PEL> prev(*this);
		++(*this);
		return prev;
}

/****************************************************************************/
// Check if iterator has more data to proccess
template <class PEL>
bool IplImageIterator<PEL>::operator!() const{
		return j < nRows; 
}

/****************************************************************************/
// Jump few pixels (advanced step must be less then image width).
// For example, use this method when you want to proccess only even pixels 
// in each line. Note when end of line is reached iterator goes to beggining of 
// new line disregarding size of the step.
template <class PEL>
IplImageIterator<PEL>& IplImageIterator<PEL>::operator+=(int s) {
	i+= s; 

	// Check if end of row is reached
	if (i >= nCols) { 
		i=start_i; 

		// Go to next line
		j++; 
		data+= step; 
	}
	return *this;
}

/****************************************************************************/
/***************************** Image Operations *****************************/
/****************************************************************************/

/****************************************************************************/
// Morphological noise removal of objects and holes of size not greater then
// minObjSize. Applies opening and then closing

void cvExtMorphologicalNoiseRemove( IplImage* img, int minObjSize, 
								    int cvShape){

	IplConvKernel* element;
	minObjSize = minObjSize/2;
    element = cvCreateStructuringElementEx( minObjSize*2+1, minObjSize*2+1, 
		                                    minObjSize,     minObjSize, 
											cvShape, NULL );

    cvErode (img,img,element,1);
    cvDilate(img,img,element,2);
    cvErode (img,img,element,1);
	cvReleaseStructuringElement(&element);	
}

/****************************************************************************/
// Morphological skeleton transform. Just a sketch

void cvExtSkeletonTransform(IplImage* srcImg, IplImage* dstImg){

	//cvCanny(fgImage8U,mask8U,10,50,3);
	//cvConvertScale(mask8U,mask8U,-1,255);
	//cvAnd(fgImage8U,mask8U,fgImage8U);
	////cvConvert(mask8U,fgImage8U);

	//cvDistTransform( mask8U, tmpImg32F, CV_DIST_L1, CV_DIST_MASK_3, NULL, NULL );
	//cvSobel(tmpImg32F,tmpImg32F,2,2,3);
	////cvLaplace(tmpImg32F,tmpImg32F,3);
	//double max;
	//cvMinMaxLoc(tmpImg32F,NULL,&max);
	//cvConvertScale( tmpImg32F, tmpImg32F, 1./max, 0);
}

/****************************************************************************/
// Invert black/white colors of binary image {0,255}

void cvExtInvertBinaryImage(IplImage* img){	
	cvConvertScale( img,img, -255, 255 );
}

/****************************************************************************/
// Strech intensity range of the image to [0..maxVal]
void cvExtIntensStrech(IplImage* img, int maxVal){
	double min, max, newScale;
	cvMinMaxLoc(img,&min,&max);
	newScale = 1.0*maxVal/(max-min);
	cvConvertScale(img,img,newScale,-min*newScale);
}
void cvExtIntensStrech(Mat& img, int maxVal){
	
	// Act with regardint to number of channels
	Mat tmpImg;
	if (img.channels() == 1){
	   tmpImg = img;
	}
	else{
	   cvtColor(img,tmpImg,CV_BGR2GRAY);
	}

	double minV, maxV,newScale;
	minMaxLoc(tmpImg,&minV,&maxV);
	newScale = 1.0*maxV/(maxV-minV);
	img = (img-minV)*newScale;
	return;
}

/****************************************************************************/
// Crops intensity to range [min..max]. 
// For each pixel, if (pix<min) it becomes min. Same for max
void cvExtIntensCrop(IplImage* img, int minVal, int maxVal){
	cvMinS(img,maxVal,img);
	cvMaxS(img,minVal,img);
}

/****************************************************************************/
// Convert mask with values in {0,1} to binary image with values {0,255}.
// Used for debugging
void cvExtConvertMask2Binary(IplImage* img){
	cvConvertScale(img,img, 0, 255 );
}

/****************************************************************************/
// Convert hue value to RGB scalar
CvScalar cvExtHue2rgb(float hue){

    int rgb[3], p, sector;
    static const int sector_data[][3]=
        {{0,2,1}, {1,2,0}, {1,0,2}, {2,0,1}, {2,1,0}, {0,1,2}};

    hue *= 0.033333333333333333333333333333333f;
    sector = cvFloor(hue);
    p = cvRound(255*(hue - sector));
    p ^= (sector&1)?255:0;

    rgb[sector_data[sector][0]] = 255;
    rgb[sector_data[sector][1]] = 0;
    rgb[sector_data[sector][2]] = p;

    return cvScalar(rgb[2], rgb[1], rgb[0],0);
}

/****************************************************************************/
// If Null then does nothing. Else acts like cvReleaseImage() and sets the 
// input image to NULL
void cvExtReleaseImageCheckNULL(IplImage** image){

	if (!*image)
	   return;

	IplImage* img = *image;
    *image = NULL;
    
    cvReleaseData(img);
    cvReleaseImageHeader(&img);
}

/****************************************************************************/
// Show image on full screen
void imshowFullScreen( const string& winname, const Mat& mat ){
	// Determine the size of the screen.
	static int cxScreen = GetSystemMetrics(SM_CXSCREEN); 
	static int cyScreen = GetSystemMetrics(SM_CYSCREEN); 
	
	// Check if window exists. If not open it in full screen.
	HWND win_handle = FindWindowA(NULL,(winname.c_str())); // A stands for ascii name of window, not unicode
	if (!win_handle){   
		cvNamedWindow(winname.c_str(),CV_WINDOW_AUTOSIZE);//,CV_WINDOW_NORMAL);
		cvMoveWindow(winname.c_str(),0,0);  // Move window to origin if needed
		win_handle = FindWindowA(NULL,(winname.c_str()));	

		// Make window full screen
	    //cvSetWindowProperty(winname.c_str(), CV_WINDOW_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		unsigned int flags = (SWP_SHOWWINDOW | SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER); flags &= ~SWP_NOSIZE; 		
		SetWindowPos(win_handle, HWND_NOTOPMOST, 0, 0, cxScreen, cyScreen, flags);  

		// Make window borderless and topmost. Do it each function call since other windows may become on top
		SetWindowLong(win_handle, GWL_STYLE, GetWindowLong(win_handle, GWL_EXSTYLE) | WS_EX_TOPMOST); 
	}

	// Show image in it.
	Mat tmp;
	resize(mat,tmp,Size(cxScreen,cyScreen));
	imshow(winname,tmp);
	ShowWindow(win_handle, SW_SHOW);
	return;
}

/****************************************************************************/
void cvShowMessageBox(const char* caption, const char* message){
	MessageBox(NULL,message,caption, MB_OK);
}

/****************************************************************************/
/*************************** Rectangle Operations ***************************/
/****************************************************************************/

/****************************************************************************/
// Returns 0  if two input rectangles are equal
// Returns +1 if jRect is inside iRect
// Returns -1 if iRect is inside jRect
// If none of this then returns -2;
int cvExtIsRectInside(CvRect iRect, CvRect jRect){
	
	if ((iRect.x == jRect.x)&&
		(iRect.y == jRect.y)&&
		(iRect.height == jRect.height)&&
		(iRect.width  == jRect.width)){
	   return 0;
	}

	// Check if 'j' is in 'i'
	if ((iRect.x <= jRect.x)&&
		(iRect.y <= jRect.y)&&
		(iRect.y + iRect.height >= jRect.y + jRect.height)&&
		(iRect.x + iRect.width  >= jRect.x + jRect.width)){
	   return +1;
	}

	// Check if 'i' is in 'j'
	if ((iRect.x >= jRect.x)&&
		(iRect.y >= jRect.y)&&
		(iRect.y + iRect.height <= jRect.y + jRect.height)&&
		(iRect.x + iRect.width  <= jRect.x + jRect.width)){
	   return -1;
	}

	return -2;
}

/****************************************************************************/
// Returns 0  if small intersection or not at all
// Returns 1  if large intersection

int cvExtIsRectLargeIntersection(CvRect iRect, CvRect jRect){
	
	double fraction = 0.15;

	iRect.x		 += cvRound(iRect.width  *fraction  );
	iRect.y		 += cvRound(iRect.height *fraction  );
	iRect.width  -= cvRound(iRect.width  *fraction*2);
	iRect.height -= cvRound(iRect.height *fraction*2);

	jRect.x		 += cvRound(jRect.width  *fraction  );
	jRect.y		 += cvRound(jRect.height *fraction  );
	jRect.width  -= cvRound(jRect.width  *fraction*2);
	jRect.height -= cvRound(jRect.height *fraction*2);

	int xIntr = ((iRect.x <= jRect.x)&&( jRect.x <= iRect.x+iRect.width ))||
				((jRect.x <= iRect.x)&&( iRect.x <= jRect.x+jRect.width ));

	int yIntr = ((iRect.y <= jRect.y)&&( jRect.y <= iRect.y+iRect.height))||
				((jRect.y <= iRect.y)&&( iRect.y <= jRect.y+jRect.height));

    return (xIntr&&yIntr);
}

/****************************************************************************/
// Returns the angle of vector from center of iRect to center of jRect
// Returns also the magnitude of the vector in last parameter
double cvExtAngBetweenRectCenters(CvRect iRect, CvRect jRect, double* vecMagnitude){

	double dX = jRect.x - iRect.x + (jRect.width  - iRect.width )/2.0;
	double dY = jRect.y - iRect.y + (jRect.height - iRect.height)/2.0;

	if (vecMagnitude) *vecMagnitude = sqrt(dX*dX+dY*dY);
			  
	return atan2(dY,dX)*180/CV_PI;
}

/****************************************************************************/
// Add two 2d vectors. Calculates 'vec' + 'd' into 'vec' arguments.
// Angles are given in degrees
void cvExtAddVector(double* vecMag, double* vecAng, double dMag, double dAng){

	*vecAng *= CV_PI/180.0;
	dAng    *= CV_PI/180.0;

	double X = (*vecMag)*cos(*vecAng) + dMag*cos(dAng);
	double Y = (*vecMag)*sin(*vecAng) + dMag*sin(dAng);
	
	*vecMag = sqrt(X*X+Y*Y)/2;			  
	*vecAng = atan2(Y,X)*180/CV_PI;
}

/****************************************************************************/
// Returns absolute difference between two angles (in degrees)

double cvExtAngAbsDiff(double ang1, double ang2 ){
	double absDiff = abs(ang1-ang2);
	return (absDiff<180)?absDiff:360-absDiff;
}

/****************************************************************************/
// Finds largest inscribed rectangle (of image proportions) inside mask.
// Returns: (x,y) uper left point corner and (lx,ly) which are width & height.
void LargestInscribedImage(const Mat& mask, Size searchImSize, int& x, int& y, int& lx, int& ly ){

	// Resize the mask to square, and reduce its size for faster running time. 
	Size s(mask.size());
	int width  = s.width;
	int height = s.height;

	double reduceFactor = 1.0;   // Higher number causes algorithm to run faster but yield less accurate results

	// Resize image to be square
	double ratio = (double)searchImSize.width/searchImSize.height;
	bool   isHorizontalImage = ratio<1?false:true;
	Size workSize(0,0);
	if (isHorizontalImage)
	   workSize = Size(cvRound(width/(ratio*reduceFactor)),cvRound(height/(reduceFactor)));
	else
	   workSize = Size(cvRound(width/(reduceFactor)),cvRound(height*ratio/(reduceFactor)));
	   
	Mat newMask;
	resize(mask,newMask,workSize);

	// Find inscribed square by
	int l;
	LargestInscribedSquare(newMask,&x,&y,&l,false);  // False = dist transform

	// Convert coordinates back to original image and round the coordinates to integers
	x  = cvRound(x*reduceFactor);
	y  = cvRound(y*reduceFactor);
	lx = cvRound(l*reduceFactor);
	ly = cvRound(l*reduceFactor);

	if (isHorizontalImage){
	   x  = cvRound(x* (1.0*ratio));  // Strech x
	   lx = cvRound(lx*(1.0*ratio));  // Strech x
	}
	else{
	   y  = cvRound(y* (1.0/ratio));  // Strech y
	   ly = cvRound(ly*(1.0/ratio));  // Strech y
	}
}

/****************************************************************************/
// Caluclates largest inscribed image allong the main diagonal using dynamic 
// programming algorithm. It is assit method to "LargestInscribedSquare"
void LargestInscribedSquareDPdiag(const Mat& mask, int *x, int*y, int *l){
	
	Size imSize(mask.size());

	// Dynamic programming tables. 
	// Changed from type CV_64F and at<double> to CV_16U and at<__int16>
	Mat Ytable(imSize.height,imSize.width,CV_16U);	
	Mat Xtable(imSize.height,imSize.width,CV_16U);	
	Mat Ltable(imSize.height,imSize.width,CV_16U);	

	// ----------------------------
	// Initialize first col	
	//Mat firstCol = mask.col(0);
	for (int i = 0; i < imSize.height; i++){
		if (mask.at<uchar>(i,0) > 0){
		   // Square can start here (current length of diagonal is 0 = one pixel)
		   Ytable.at<__int16>(i,0) = i;
		   Xtable.at<__int16>(i,0) = 0;
		   Ltable.at<__int16>(i,0) = 0;
		}
		else{
		   Ytable.at<__int16>(i,0) = 0;
		   Xtable.at<__int16>(i,0) = 0;
		   Ltable.at<__int16>(i,0) = -1;
		}
	}

	// ----------------------------
	// Initialize first row	
	for (int i = 0; i < imSize.width; i++){
		if (mask.at<uchar>(0,i) > 0){
		   // Square can start here (current length of diagonal is 0 = one pixel)
		   Ytable.at<__int16>(0,i) = 0;
		   Xtable.at<__int16>(0,i) = i;
		   Ltable.at<__int16>(0,i) = 0;
		}
		else{
		   Ytable.at<__int16>(0,i) = 0;
		   Xtable.at<__int16>(0,i) = 0;
		   Ltable.at<__int16>(0,i) = -1;
		}
	}

	// ----------------------------
	//% Dynamic Programming. (from (1,1) to end of matrix)
	for (int i = 1; i < imSize.height; i++){
		//const char* mask_i = mask.ptr<char>(i);
		for (int j = 1; j < imSize.width; j++){

			if (mask.at<uchar>(i,j) == 0){
				// No square including this pixel, just forward information
				Ytable.at<__int16>(i,j) = Ytable.at<__int16>(i-1,j-1);
				Xtable.at<__int16>(i,j) = Xtable.at<__int16>(i-1,j-1);
				Ltable.at<__int16>(i,j) = Ltable.at<__int16>(i-1,j-1);
			}
			else{
				if (mask.at<uchar>(i-1,j-1) > 0){

 				   Ytable.at<__int16>(i,j) = Ytable.at<__int16>(i-1,j-1);
		 		   Xtable.at<__int16>(i,j) = Xtable.at<__int16>(i-1,j-1);

				   // It is a continuous diagonal. Chack secondary diagonal
				   if ((mask.at<uchar>(Ytable.at<__int16>(i,j),j) > 0)&&
					   (mask.at<uchar>(i,Xtable.at<__int16>(i,j)) > 0)){
				      // Square can be enlarged 
				      Ltable.at<__int16>(i,j) = Ltable.at<__int16>(i-1,j-1)+1;
				   }
				   else{
					  // Square cannot be extended. Just forward the information
					  Ltable.at<__int16>(i,j) = Ltable.at<__int16>(i-1,j-1);
				   }
				}
				else{
				   // This pixel starts a square 
				   Ytable.at<__int16>(i,j) = i;
				   Xtable.at<__int16>(i,j) = j;
				   Ltable.at<__int16>(i,j) = 0;
				}				   
			}
		}
	}

	// ----------------------------
	// Find max L value in last column
	int maxValCol = 0,WhereCol = 0;
	for (int i = 0; i < imSize.height; i++){
		int l = Ltable.at<__int16>(i,imSize.width-1);
		if (l>maxValCol){
		   maxValCol = l;
		   WhereCol  = i;
		}
	}

	// Find max L value in last row
	int maxValRow = 0,WhereRow = 0;
	for (int i = 0; i < imSize.width; i++){
		int l = Ltable.at<__int16>(imSize.height-1,i);
		if (l>maxValRow){
		   maxValRow = l;
		   WhereRow  = i;
		}
	}		

	// Take coordinates with largest L
	if (maxValCol>maxValRow){
	   *x = Xtable.at<__int16>(WhereCol,imSize.width-1);
	   *y = Ytable.at<__int16>(WhereCol,imSize.width-1);
	   *l = maxValCol;
	}
	else {
	   *x = Xtable.at<__int16>(imSize.height-1,WhereRow);
	   *y = Ytable.at<__int16>(imSize.height-1,WhereRow);
	   *l = maxValRow;
	}
	return;
}

/****************************************************************************/
// Input mask (white pixels are the enclosing area, blacks are background)
// Mask is char (8U) Mat with white > 0, black is = 0. 
// Returns: square location (x,y)-> (x+l,y+l)
// If mask covers convex area then faster algorithm will be launged.
void LargestInscribedSquare  (const Mat& mask, int *x, int*y, int *l, bool isMaskConvex){
	
	if (!isMaskConvex){     // Use slow distance transform algorithm
		Mat res;
		
		// Make zero padding so mask will not tuch image borders.  We loose here about 0.5 pix.
		mask.row(0) = 0;            // Use copyMakeBorder() instead
		mask.col(0) = 0;
		mask.row(mask.rows-1) = 0;
		mask.col(mask.cols-1) = 0;

		// Calculate distance transform and retrieve values
		distanceTransform(mask, res, CV_DIST_C, CV_DIST_MASK_5);
		//debugShowImage(res/40);
		double minVal, maxVal;
		Point maxLoc(0,0);
		minMaxLoc(res, &minVal,&maxVal, 0,&maxLoc);
		*x = cvRound(maxLoc.x - maxVal);
		*y = cvRound(maxLoc.y - maxVal);
		*l = cvRound(maxVal*2+1);
	}
	else {				  // Use dynamic programming faster algorithm for convex masks
		// Find inscribed square by main diagonal
		int x1,y1,l1;
		LargestInscribedSquareDPdiag(mask,&x1,&y1,&l1);

		// Find inscribed square by secondary diagonal. Make that using the same algorithm but rotate 90 deg backward clock direction.
		int x2,y2,l2;
		Mat newMask = mask.t();
		flip( newMask, newMask,  0 ); 
		LargestInscribedSquareDPdiag(newMask,&x2,&y2,&l2);

		// Convert the rotated x,y to original coordinates
		int tmpInt = x2;
		x2 = newMask.size().height - y2 - l2;
		y2 = tmpInt;

		// Calc biggest square
		if (l1 > l2){
			*x = x1; *y = y1; *l = l1;
		}
		else{
  			*x = x2; *y = y2; *l = l2;
		}
	}
}

/****************************************************************************/
// Finds the corners of a given mask. Returns them in clockwise order in corners vector.
// Mask is an arbitrary quadrilateral white shape on black background.
// Best value for distThr is ~ 0.8 of mask smallest side.
// Returns the area of the detected quadrilateral
double findCornersOfMask(Mat& maskImg, int numCorners, vector<Point2f>& corners, double distThr){		

	// Extract the corners.	
    vector<Point2f> tmpCorners;
	goodFeaturesToTrack( maskImg, tmpCorners, numCorners, 0.1,distThr, NULL_MATRIX, 9);    	

	// Calculate convex hull in order to remove wrong corners inside the mask. 
	// Arrange convex hull clockwise
	vector<Point2f> convexCORN;
	if (tmpCorners.empty()){
	   corners.clear();
	   return 0; // No corners were detected 
	}
	   
	convexHull(Mat(tmpCorners),convexCORN,false);

	// Find the upper left point in the convex hull.
	float minDist = (float)maskImg.cols*maskImg.cols + maskImg.rows*maskImg.rows;
	int index     = -1;
	numCorners     = (int)convexCORN.size();
	for (int i = 0; i < numCorners; i++){
		float curDist = convexCORN[i].x*convexCORN[i].x+convexCORN[i].y*convexCORN[i].y;
		if (curDist < minDist){
			minDist = curDist;
			index   = i;			
		}
	}

	// Rearrange the points so upper left one will be first. Keep clockwise order
	for (int i=0; i<numCorners; ++i){
		corners.push_back(convexCORN[(i+index)%numCorners]);
	}

	// Calculate area of convex hull	
	double area = cvExtConvexHullArea(corners);
	return area;
}

/****************************************************************************/
// Finds the area of 2D convex hull (ordered vector of points).
double cvExtConvexHullArea(vector<Point2f>& ch){
	double area = 0;
	for (unsigned int i = 0; i < ch.size(); i++){
		int next_i = (i+1)%(ch.size());
		double dX   = ch[next_i].x - ch[i].x;
		double avgY = (ch[next_i].y + ch[i].y)/2;
		area += dX*avgY;  // This is the integration step.
	}
	return fabs(area);

	// Different implementation
/*    vector<Point2f> contour; 
    approxPolyDP(Mat(ch), contour, 0.001, true);
	return fabs(contourArea(Mat(contour)));*/

}

/****************************************************************************/
/****************************** Image matching ******************************/
/****************************************************************************/

/****************************************************************************/
// Uses built in matching method for cross check matching. Matches src->dst, dst->src
// and findes bi-directional matches.
void cvExtCrossCheckMatching( DescriptorMatcher* matcher, const Mat& descriptors1, const Mat& descriptors2,
							 vector<DMatch>& matches, int knn ){
    
	// Calculate bi-directional matching						 
	matches.clear();
    vector<vector<DMatch>> matches12, matches21;
    matcher->knnMatch( descriptors1, descriptors2, matches12, knn );
    matcher->knnMatch( descriptors2, descriptors1, matches21, knn );

	// Iterate over 1->2 matchings
    for( size_t m = 0; m < matches12.size(); m++ ){
        bool findCrossCheck = false;
		// For current match, iterate over its N best results.
        for( size_t fk = 0; fk < matches12[m].size(); fk++ ){
            DMatch forward = matches12[m][fk];
			// For each result iterate over its best backward matching
            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ ){
                DMatch backward = matches21[forward.trainIdx][bk];
				// if this match is the same as one of backward matches of one ogf its N results then we have found a bi-directional match.
                if( backward.trainIdx == forward.queryIdx ){
                    matches.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if (findCrossCheck) 
			   break;
        }
    }
}

/****************************************************************************/
/*************************** Drawing on images ******************************/
/****************************************************************************/

/****************************************************************************/
// Draw arrow on the image (img) in a given location (start) in direction (angDeg),
// with length (length). Color and thikcness are also parameters.

void cvExtDrawArrow(IplImage *img, CvPoint start, double angDeg, double length,
					               CvScalar color, int thickness){

	CvPoint endPoint, arrowPoint;

    // Convert to radians
	double angle = angDeg*CV_PI/180.0;

	// Find end point of the arrows body.
	endPoint.x = (int)(start.x + length*cos(angle));
	endPoint.y = (int)(start.y + length*sin(angle));

	// Draw the body (main line) of the arrow.
	cvLine(img, start, endPoint, color, thickness, CV_AA, 0);

	// Draw the tips of the arrow, scaled to the size of the main part
	double tipsLength = 0.4*length;
	arrowPoint.x = (int)(endPoint.x + tipsLength*cos(angle + 3*CV_PI/4));
	arrowPoint.y = (int)(endPoint.y + tipsLength*sin(angle + 3*CV_PI/4));
	cvLine(img, arrowPoint, endPoint, color, thickness, CV_AA, 0);

	arrowPoint.x = (int) (endPoint.x + tipsLength*cos(angle - 3*CV_PI/4));
	arrowPoint.y = (int) (endPoint.y + tipsLength*sin(angle - 3*CV_PI/4));
	cvLine(img, arrowPoint, endPoint, color, thickness, CV_AA, 0);

	// Draw circle around the arrow.
	// cvCircle(img, start, cvRound(length *1.2), color, thickness, CV_AA, 0 );
}

/****************************************************************************/
// Draw Object rectangle with an arrow from his center pointing in defined 
// direction. Used for drawing global motion direction in swordfish debug mode.
void cvExtDrawObj(IplImage *img, CvRect objRect, double angDeg, double length, 
				  CvScalar color){

        // draw rectangle in the image
        cvRectangle(img, cvPoint(objRect.x, objRect.y),
                    cvPoint(objRect.x+objRect.width, objRect.y+objRect.height),
                    color, 2+0*CV_FILLED, 8);
    
        // Draw a clock with arrow indicating the direction
        CvPoint center = cvPoint((objRect.x + objRect.width/2), 
			                     (objRect.y + objRect.height/2));

		cvExtDrawArrow(img, center, angDeg, length, color);
}

/****************************************************************************/
// Visualizes a histogram as a bar graph into an output image.
// Uses different Hue for each bar.
void cvExtVisualizeHueHistogram(const CvHistogram *hist, int hdims, IplImage *img){

    // Temporary color
	CvScalar color = cvScalar(0,0,0,0);
	
	cvZero(img);

	// Calculate width of a bin
    int bin_w = img->width/hdims;

	// Iterate over all bins
    for( int i = 0; i < hdims; i++ ){
        int val = cvRound( cvGetReal1D(hist->bins,i)*(img->height)/255 );

		// Attach new hue for each bin
        color = cvExtHue2rgb((i*180.f)/hdims);

		// Draw the bin
        cvRectangle( img, cvPoint(i*bin_w,img->height),
                     cvPoint((i+1)*bin_w,img->height - val),
                     color, CV_FILLED , 8, 0 );
    }
}

/****************************************************************************/

// Calculates Pow 2 of absolute value of edges using soble edge detector. 
// All images must be preallocated. 
// auxImage and srcImages are changed in the function (not const)
void cvExtCalcEdgesPow2( IplImage* edgeImage, IplImage* srcImage, IplImage* auxImage){

	// Calculate dX and dY of gray image
	cvCopy(srcImage,auxImage);
	cvSobel(srcImage,srcImage,1,0,3);
	cvSobel(auxImage,auxImage,0,1,3);

	// Calculate absolute edge strength: dX^2+dY^2.
	cvMul(srcImage,srcImage,srcImage);
	cvMul(auxImage,auxImage,auxImage);
	cvAdd(srcImage,auxImage,edgeImage);	
	return;
}


/****************************************************************************/
// Reduces image brightness to nBit representation (2^nBits levels).
void cvExtReduceTonBits(IplImage* srcImage, unsigned int nBits){
     nBits = (nBits>8)?8:nBits;
	 if (nBits<8){
	   int depthRatio = cvRound(256/pow(2.0,1.0*nBits));
	   int multFactor = (depthRatio!=256)?255/(256/depthRatio -1):0;
	   cvExtImageIterator it(srcImage);
	   while (!it){
			*it = cvRound(*it/depthRatio)*multFactor;
			++it;
	   }
	 }
	 return;
}

/****************************************************************************/
// Reduces number of levels in the image to nLeves.
// Image must be given as float
void cvExtReduceLevels(IplImage* srcImage, unsigned int nLevels,  bool smartLevels){
	
	// Check if number of levels is legal
	if ((nLevels < 2)||(nLevels>255)){
	   return;
	}

	// Check if image type is legal
	if (srcImage->depth != IPL_DEPTH_32F){
	   cvError(1,"cvExtReduceLevels","wrong image argument","cvExtension",1);
	   return;
	}

	cvExtIntensStrech(srcImage,255); // Strech image to [0..255]

	// If levels are just chosen uniformly, make it and quit the method
	if (!smartLevels){
	   int depthRatio = cvRound(256/nLevels);
	   int multFactor = (depthRatio!=256)?255/(256/depthRatio -1):0;
	   IplImageIterator<float> it(srcImage);
	   while (!it){
			*it = (float)cvRound(*it/depthRatio)*multFactor;
			++it;
	   }
	   return;
	}
    IplImage* AuxImage   = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_32F, 1);

    // Calculate nLevels-means of srcImage to auxillary matrix
    int numPixels = srcImage->height*srcImage->width*srcImage->nChannels;
    CvMat points         = cvMat(numPixels, 1, CV_32FC1, reinterpret_cast<float*>(srcImage->imageData));
    CvMat clusters       = cvMat(numPixels, 1, CV_32SC1, reinterpret_cast<float*>(AuxImage->imageData));
    cvKMeans2( &points, nLevels, &clusters, cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ));

    // Convert the auxillary matrix to AuxImage
    CvMat row_header, *row;
    row = cvReshape( &clusters, &row_header, 0, srcImage->height );
    AuxImage = cvGetImage(row,AuxImage);

    // AuxImage includes for each pixel its cluster (class)
	// Let us calculate for each class its true color. It would be average of brightness of 
	// all pixels that belong to the current class

    float* numPixelsInClass   = new float[nLevels];
    float* avgClassBrightness = new float[nLevels];
    for (unsigned int i=0; i<nLevels; i++){
	   numPixelsInClass  [i] = 0;
	   avgClassBrightness[i] = 0;
    }

    cvExt32FImageIterator origPix   (srcImage);
    IplImageIterator<int> clusterPix(AuxImage);  // Cluster is integer
    while (!clusterPix){
	   avgClassBrightness[*clusterPix] += *origPix;
	   numPixelsInClass  [*clusterPix]++;
	   ++origPix;
	   ++clusterPix;
    }

	// Calculate the average
    for (unsigned int i=0; i<nLevels; i++){
	    avgClassBrightness[i] /= numPixelsInClass[i];
    }

	// Convert Auxillary image from pixels of classes to pixels of true colora
    IplImageIterator<int> result(AuxImage);
    while (!result){
	   *result = cvRound(avgClassBrightness[*result]);
	   ++result;
    }
 
	// Insert the result to srcImage and strech intensity to [0..255]
	cvConvertScale(AuxImage,srcImage);
	//cvExtIntensStrech(srcImage,255);

	// Delete all auxillary memory
    delete[] avgClassBrightness;
    delete[] numPixelsInClass;
    cvExtReleaseImageCheckNULL(&AuxImage);

	return;
}

/****************************************************************************/
// Convert image to pencil sketch
void cvExtPhoto2Pencil(const IplImage* srcImage, IplImage* dstImage, 
					   double intensNormal,
					   int    looseDitails,
					   int    pencilLines,
					   int colorNormal, int colorBits, int fast){

	// -----------------------------------------
	// Auxillary images
	IplImage* gryImage  = cvCreateImage( cvGetSize(srcImage), IPL_DEPTH_8U , 1 );
	IplImage* hueImage  = cvCreateImage( cvGetSize(srcImage), IPL_DEPTH_8U , 1 );
	IplImage* satImage  = cvCreateImage( cvGetSize(srcImage), IPL_DEPTH_8U , 1 );
	IplImage* edgImage  = cvCreateImage( cvGetSize(srcImage), IPL_DEPTH_32F, 1 );
	IplImage* tmp1Image = cvCreateImage( cvGetSize(srcImage), IPL_DEPTH_32F, 1 );
	IplImage* tmp2Image = cvCreateImage( cvGetSize(srcImage), IPL_DEPTH_32F, 1 );

	// Start Algorithm
	cvCopy(srcImage,dstImage);

	// -----------------------------------------
	// Loose ditails in the image. 

	if (looseDitails < 3)
	   for (int i=0; i<looseDitails;i++){
	   	   cvSmooth(dstImage,dstImage,CV_MEDIAN,3);
	   }
	   // cvSmooth(dstImage,dstImage,CV_MEDIAN,1+2*looseDitails);
	else{
	   // Loose details due to morphologic closure
	   int winS = looseDitails-1;
	   IplConvKernel* element;
	   element = cvCreateStructuringElementEx( winS*2+1, winS*2+1, winS, winS, CV_SHAPE_ELLIPSE, 0 );
	   cvErode (dstImage,dstImage,element,1);
	   cvDilate(dstImage,dstImage,element,1);  
	}

	// -----------------------------------------
	// Convert image to HSV
	cvCvtColor( dstImage, dstImage, CV_BGR2HSV );
	cvSplit( dstImage, hueImage, satImage, gryImage, NULL);

	// -----------------------------------------
    // Put black lines on edges

	// Calculate dX^2+dY^2 of gray image (intensity)
    // We dont do sqrt() on edge strength since we will include it in gamma correction
	cvConvertScale(gryImage,tmp1Image,1.0/255,0);
	cvExtCalcEdgesPow2(edgImage,tmp1Image,tmp2Image);

	// Calculate dX^2+dY^2 of saturation image
	cvConvertScale(satImage,tmp1Image,1.0/255,0);
	cvExtCalcEdgesPow2(tmp1Image,tmp1Image,tmp2Image);
	cvConvertScale(gryImage,tmp2Image,1.0/255,0);
    cvMul(tmp1Image,tmp2Image,tmp1Image);       // Reduce saturation edges in dark areas of the image

	// Take the maximum of both edges
	cvMax(tmp1Image,edgImage,edgImage);
	cvExtIntensStrech(edgImage,1);    

	// Invert the edge (in order to turn it black), and change its gamma
	int gamma = 6*(pencilLines+4);
	cvExt32FImageIterator edge(edgImage);
	while (!edge){
		*edge = pow((1-(*edge)),gamma);
		++edge;
	}

	// -----------------------------------------
	// Normalize intensity variations in the image.

	if (fast){
	   // by subtracting from the image its blurred background.
	   // Formulae: (a)*Img + (1-a)*(Inverted bloored image). a is in [0.5..1].
	   // Note: when 'a' is higher then 1 the normalization is inversed

	   cvConvertScale(gryImage,tmp1Image,1/255.0,0);
	   cvSmooth(gryImage,gryImage,CV_BLUR,5,5);
	   cvConvertScale(gryImage,tmp2Image,-1/255.0,1);

	   double iN = 0.5 + 0.6*intensNormal;
	   cvAddWeighted(tmp1Image,iN,tmp2Image,1-iN,0,tmp1Image);
	} 
	else {
	   // Find best intensity levels using smart Reduction of levels.
	   // Add 2 to levels in order to have minimum levels of 2.
	   cvConvertScale(gryImage,tmp1Image);
	   cvExtReduceLevels(tmp1Image,cvRound((intensNormal)*10+2));
	   cvConvertScale(tmp1Image,gryImage);
	   cvExtIntensStrech(gryImage,255);
	   cvSmooth(gryImage,gryImage,CV_BLUR,5,5);
	   cvConvertScale(gryImage,tmp1Image,1/255.0);
	}

	// -----------------------------------------
	// Combine all information (edges and normalized image) into final intensity value

	cvMul(tmp1Image,edgImage,tmp1Image);
	cvExtIntensStrech(tmp1Image,1);
	cvConvertScale(tmp1Image,gryImage,255.0,0);


	// -----------------------------------------
	// Reduce The saturation in dark areas of original image for removing artifacts
	// Do this by: Sat = Sat * (Gamma corrected Blurred Background) * colorNormal
	// By the way, Inverted Blurred background is stored in tmp2Image

	cvConvertScale(satImage,tmp1Image,1/255.0,0);

	// Correct Gamma of blurred background (use gamma > 1 since it is inverted)
	// Also do 1-result to invert the background back

	int gamma2 = colorNormal/2;
	cvConvertScale(gryImage,tmp2Image,-1/255.0,1);
	cvExt32FImageIterator back(tmp2Image);
	while (!back){
		*back = 1-pow(*back,gamma2);
		++back;
	}

	double colorMultiply = (colorNormal<5)?(0.2+colorNormal/5.):(1+colorNormal/20.);
	cvMul(tmp1Image,tmp2Image,tmp1Image,colorMultiply);
	cvExtIntensCrop(tmp1Image,0,1);
	cvConvertScale(tmp1Image,satImage,255.0,0);

	// -----------------------------------------
	// Reduce accuracy of saturation (reduce number of levels and impose smoothness)

	if (fast){
	   cvExtReduceTonBits(satImage,colorBits);
	}
	else {
	   cvExtReduceLevels(tmp1Image,colorBits+2);
	   cvConvertScale(tmp1Image,satImage);
	}

	int longest    = CVEXT_MAX(satImage->height,satImage->width);
	cvSmooth(satImage,satImage,CV_BLUR,cvRound(longest/200.0)*2+1);

	// -----------------------------------------
	//  Alter the hue of the image. Reduce number of colors in the image
	//  Or even convert entire image to single color.

/*	cvConvertScale(hueImage,tmp1Image);
	cvExtReduceLevels(tmp1Image,cvRound(1+pow(1.4,colorBits)));
	cvConvertScale(tmp1Image,hueImage,180.0/255.0);    */

	// -----------------------------------------
	// Incorporate back into image & Return to RGB plane
    cvMerge(hueImage, satImage, gryImage, NULL, dstImage );
	cvCvtColor(dstImage, dstImage, CV_HSV2BGR);

	// -----------------------------------------
	// Free allocated images
    cvExtReleaseImageCheckNULL(&gryImage);
    cvExtReleaseImageCheckNULL(&hueImage);
    cvExtReleaseImageCheckNULL(&satImage);
    cvExtReleaseImageCheckNULL(&edgImage);
    cvExtReleaseImageCheckNULL(&tmp1Image);
    cvExtReleaseImageCheckNULL(&tmp2Image);
}

/*
	// -----------------------------------------
	// Light pencil stroke matrix. (Sum of the matrix must be 1)!
	double _pencilStroke[] = {0,0,0,2,1,0,0,
							  0,0,2,3,2,1,0,
							  0,2,4,5,4,2,1,
							  2,3,5,7,5,3,2,
							  1,2,4,5,4,2,0,
							  0,1,2,3,2,0,0,
							  0,0,1,2,0,0,0};

	CvMat pencilStroke = cvMat( 7, 7, CV_64F, _pencilStroke );
	CvScalar normStroke = cvSum(&pencilStroke);
	cvConvertScale(&pencilStroke,&pencilStroke,1/(normStroke.val[0]),0);
	
	// Will make the strokes random
	static CvRNG rng = cvRNG(-1);

	// -----------------------------------------
	// Add random pencil strokes (above the edge image)
	cvRandArr( &rng, tmp1Image, CV_RAND_UNI, cvScalarAll(-1), cvScalarAll(1));
	cvFilter2D(tmp1Image,tmp1Image,&pencilStroke);
	cvSmooth(tmp1Image,tmp1Image,CV_GAUSSIAN,3,3);

	cvAddWeighted(edgImage,1,tmp1Image,lightPencil,0,edgImage);
	cvExtIntensCrop(edgImage);
*/

/****************************************************************************/
// I/O
/****************************************************************************/

// Check if output directory exists. If not, create it. Returns true if OK.
bool createDirIfNotExists(char *DirPath){
	if (!DirPath){
	   return false;
	}

	// Check if output directory exists.
	DWORD dwAttr = GetFileAttributesA(DirPath);
	if (!(dwAttr != 0xffffffff && (dwAttr & FILE_ATTRIBUTE_DIRECTORY))){
	   return CreateDirectoryA(DirPath,NULL) != 0;
	}
	// Already exists
	return true;
}

/****************************************************************************/

bool isFileExists(char *filePath){
	if (!filePath){
	   return false;
	}
	DWORD dwAttr = GetFileAttributesA(filePath);
	return (dwAttr != 0xffffffff);
}
/****************************************************************************/

/****************************************************************************/
/************************* Video Frame Grabber ******************************/
/****************************************************************************/

/****************************************************************************/
// Detect which type of media is stored in the fiven file path

cvExtMediaType cvExtDetectMediaType(const string path){
	const char *extention = &path[path.length()-3];

	// Suported text sequences
	if (!_stricmp(extention,"txt")){
	   return FRAMES_TEXT_LIST;  // 
	}

	// Suported video types
	if ((!_stricmp(extention,"avi"))||
		(!_stricmp(extention,"mpg"))||
		(!_stricmp(extention,"wmv"))){
	   return OFFLINE_VIDEO;  
	}

	// Suported image types
	if ((!_stricmp(extention,"jpg"))||
		(!_stricmp(extention,"tif"))||
		(!_stricmp(extention,"png"))||
		(!_stricmp(extention,"bmp"))||
		(!_stricmp(extention,"gif"))){
	   return SINGLE_IMAGE;
	}

	// Un-Suported video types
	if ((!_stricmp(extention,"mov"))||
		(!_stricmp(extention,"mp3"))||
		(!_stricmp(extention,"mp4"))){
	   return UNKNOWN;  
	}

	// Supported online displays
	if (!_stricmp(extention-1,"@mov")){
	   return ONLINE_WINDOW;  
	}

	return UNKNOWN;
}

/****************************************************************************/
// Empty Constructor for online grabbing (from camera)
cvExtFrameGrabber::cvExtFrameGrabber(int camIndex){

	frameNumber	    = 0;
	 verticalFlipFlag   = false;
	 horizontalFlipFlag = false;
	 textFile		    = NULL;
	 //curFrame		    = NULL;	  

	 if (capture.open(camIndex)){
	    framesSource    = ONLINE_CAMERA;
	    FPS             = capture.get(CV_CAP_PROP_FPS);
	    codec           = cvRound(capture.get(CV_CAP_PROP_FOURCC));
	 }
	 else{
	    framesSource    = UNKNOWN;
		FPS             = CVEXT_WRONG_FPS;
		codec           = CVEXT_WRONG_CODEC;
	 }

	 isUndistortActivated = false;
}

/****************************************************************************/
// Constructor with file path for offline grabbing
cvExtFrameGrabber::cvExtFrameGrabber(string path){

	 videoFilePath      = path;
	 frameNumber        = 0;
	 verticalFlipFlag   = false;
	 horizontalFlipFlag = false;
	 textFile		    = NULL;
	 //curFrame		    = NULL;
	 //capture            = NULL;
	 framesSource       = cvExtDetectMediaType(videoFilePath);

	 switch (framesSource){
	 case OFFLINE_VIDEO:
		 if (capture.open(videoFilePath)){
			FPS          = capture.get(CV_CAP_PROP_FPS);
			codec        = cvRound(capture.get(CV_CAP_PROP_FOURCC));
		 }
		 else{
		    framesSource = UNKNOWN;
		    FPS          = CVEXT_WRONG_FPS;
		    codec        = CVEXT_WRONG_CODEC;
		 }
  		 break;


 	 case FRAMES_TEXT_LIST:	
 	     // Check if this a text file which stores list of images
		 fopen_s(&textFile,videoFilePath.c_str(), "rt" );
		 if (!textFile){
 		    framesSource = UNKNOWN;
		 }

		 // FPS and codec are irrelevant
	     FPS             = CVEXT_WRONG_FPS;
	     codec           = CVEXT_WRONG_CODEC;
  		 break;

 	 case SINGLE_IMAGE:	
		 // FPS and codec are irrelevant
	     FPS             = CVEXT_WRONG_FPS;
	     codec           = CVEXT_WRONG_CODEC;
		 break;		 
     }

	 isUndistortActivated = false;
}

/****************************************************************************/
// Constructor for online grabbing from matrix (which can be changed from outside). Mostly used for debug
cvExtFrameGrabber::cvExtFrameGrabber(const Mat *sourceMat){

	 verticalFlipFlag   = false;
	 horizontalFlipFlag = false;
	 textFile		    = NULL;
	 this->sourceMat    = sourceMat;
     framesSource       = MAT_PTR_DEBUG;

	 isUndistortActivated = false;
}

/****************************************************************************/
// Destructor
cvExtFrameGrabber::~cvExtFrameGrabber(){
	
	// Close the grabber
	if (capture.isOpened()){
		capture.release();
	}

	// Close the text file if needed
	 if (textFile){
        fclose(textFile);
     }
}

/****************************************************************************/
// Returns current frame. User must NOT release or change the image. Returned image will dissapear
// when next frame arrives or this object is destroyied.
const Mat cvExtFrameGrabber::watchFrame(void){

	switch (framesSource){

	case ONLINE_CAMERA:
	case OFFLINE_VIDEO:
		 capture >> this->curFrame;
  		 break;

	case FRAMES_TEXT_LIST:
		 while (fgets(imageName, sizeof(imageName)-2, textFile )){

  			   int l = (int)strlen(imageName);
			   if (!l){
			      // empty line occured
			      continue;
			   }
			   if (imageName[0] == '#' ){
			      // Commented frame
				  continue;
			   }

		 	   //imageName[l-1] == '\n')
 			   imageName[--l] = '\0';

			   // Check valid extension
			   int extension = (int)strlen(imageName)-3;
			   if ((!_stricmp(&imageName[extension],"jpg"))||
			       (!_stricmp(&imageName[extension],"tif"))||
				   (!_stricmp(&imageName[extension],"png"))||
				   (!_stricmp(&imageName[extension],"bmp"))||
				   (!_stricmp(&imageName[extension],"gif"))){

			      curFrame = imread(imageName,CV_LOAD_IMAGE_COLOR);
				  break;
			   }		      			   
		 }     // End of While
  	     break;


    case SINGLE_IMAGE:	
		 // Load the image and turn source to depleted in order not to load the image again
		 curFrame = imread(videoFilePath,CV_LOAD_IMAGE_COLOR);
		 framesSource = DEPLETED;
		 break;

	case MAT_PTR_DEBUG:
		 curFrame = *(this->sourceMat);
  	     break;

	case UNKNOWN:
	case DEPLETED:
	 	 curFrame.release();
		 break;

	} //End switch

	if (!curFrame.empty()){

	   // If image loaded
	   frameNumber++;

	   // Undistort if needed
	   if (isUndistortActivated){
		  Mat tmp;
	      undistort(curFrame,tmp,IntrinsicMat, DistCoef);
		  curFrame = tmp;
	   }

	   // Flip if needed 
	   if (verticalFlipFlag)
 	      if (horizontalFlipFlag)
	         flip( curFrame, curFrame, -1 );
		  else
	         flip( curFrame, curFrame,  1 );
	   else
 	      if (horizontalFlipFlag)
	         flip( curFrame, curFrame,  0 );

	   
	}
	else {
	   framesSource = DEPLETED;
	}

	return curFrame;
}

/****************************************************************************/
// Returns current frame. User may change the image but also responsible for releasing it.
Mat cvExtFrameGrabber::grabFrame(void){
	return (this->watchFrame()).clone();
}

/****************************************************************************/
// Backward compatibility grabbing functions

const IplImage cvExtFrameGrabber::watchFrameIpl(void){
	grabFrame();
	return ((IplImage)curFrame);
}

IplImage* cvExtFrameGrabber::grabFrameIpl(void){
	grabFrame();
	IplImage tmp = (IplImage)curFrame;
	if (!curFrame.empty())
	   return cvCloneImage(&tmp);
	else
	   return NULL;
}

/****************************************************************************/
// Get Methods:

// Returns the number of current frame
int cvExtFrameGrabber::getCurFrameNumber(){
	return frameNumber;
}

// Change vertical flipping flag. If flag is set then
// All frames would be flipped vertically. 
void cvExtFrameGrabber::setVerticalFlip(bool flag){
	verticalFlipFlag = flag;
}
void cvExtFrameGrabber::setHorizontalFlip(bool flag){
	horizontalFlipFlag = flag;
}

// Return true if current video source is online (realtime)
bool cvExtFrameGrabber::isRealTimeGrabbing(void){
	return ((framesSource == ONLINE_CAMERA));//||
		    //(framesSource == MAT_PTR_DEBUG));
}

// Return true if current source is video and not sequence of images.
bool cvExtFrameGrabber::isVideoStreamGrabbing(void){
	return ((framesSource == ONLINE_CAMERA)||
		    (framesSource == OFFLINE_VIDEO));//||
			//(framesSource == MAT_PTR_DEBUG));
}

/****************************************************************************/
// Set/Get Methods:

// Return media frames per second ratio
double cvExtFrameGrabber::getFPS(void){
	return FPS;
}

// Return media decompressing codec
int cvExtFrameGrabber::getMediaCodec(void){
	return codec;
}


// Set/get size of the image.
// CvSize  cvSize( int width, int height )
bool cvExtFrameGrabber::setMediaSize(int width, int height){	
	if (!isRealTimeGrabbing()){
	   return false;
	}
	bool res = capture.set(CV_CAP_PROP_FRAME_WIDTH,width);
	if (res){
	     res = res & capture.set(CV_CAP_PROP_FRAME_HEIGHT,height);
	}
	return res;
}

/****************************************************************************/
void cvExtFrameGrabber::getMediaSize(int*width, int*height){	
	if (isRealTimeGrabbing()){
		*width  = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
		*height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	}
}

/****************************************************************************/
// Activate undistortion on the media. Distortion parameters are stored in file in camera.xml:
bool cvExtFrameGrabber::activateUndistort(const string filePath){

	 FileStorage fs(filePath, FileStorage::READ);
	 isUndistortActivated = fs.isOpened();
     if (!isUndistortActivated){
		return false;
     }

	 fs["intrCamMat"] >> IntrinsicMat;
	 fs["distortion"] >> DistCoef;
	 fs.release(); 
	 return true;
}

/****************************************************************************/
/************************* Video Frame Writer ******************************/
/****************************************************************************/

// Constructor
cvExtFrameWriter::cvExtFrameWriter(string path, double fps, int codec, bool isColored){

	 videoFilePath      = path;
	 dirEnd             = videoFilePath.rfind("\\");
	 dirEnd             = (dirEnd!=string::npos)?dirEnd:-1;

	 frameNumber        = 0;
	 verticalFlipFlag   = false;
	 horizontalFlipFlag = false;
	 textFile		    = NULL;
	 //writer             = NULL;

	 vFPS				= ( fps   != CVEXT_WRONG_FPS   ) ? fps   : 29.97;            // Default value is 29.97
	 vCodec				= ( codec != CVEXT_WRONG_CODEC ) ? codec : CV_FOURCC_PROMPT; // Default is divx
	 vIsColored			= isColored; 
	 
	 framesDest         = cvExtDetectMediaType(videoFilePath);
	 switch (framesDest){
	 case OFFLINE_VIDEO:
		 // Do nothing
  		 break;

 	 case FRAMES_TEXT_LIST:	
	     // Try to open file for writing
		 fopen_s(&textFile,videoFilePath.c_str(), "wt" );
		 framesDest = (textFile)? FRAMES_TEXT_LIST : UNKNOWN;
  		 break;

 	 case SINGLE_IMAGE:	
		 // Do nothing
		 break;	 

	 case ONLINE_WINDOW:
		 // Erase the extension
		 videoFilePath.erase(videoFilePath.length()-4,4);
  		 break;
     } // switch

}
		  //CV_FOURCC('M', 'J', 'P', 'G'), 
		  // CV_FOURCC('D', 'I', 'V', 'X')

/****************************************************************************/
// Destructor
cvExtFrameWriter::~cvExtFrameWriter(){

	 // Close the writer

	 // Close the text file if needed
	 if (textFile){
        fclose(textFile);
     }
}

/****************************************************************************/
// Writes current frame.
void cvExtFrameWriter::writeFrame(const IplImage* frame){
	this->writeFrame(Mat(frame));
}

void cvExtFrameWriter::writeFrame(const Mat& frame){

	switch (framesDest){

	case ONLINE_WINDOW:
		 imshow(videoFilePath,frame);
  		 break;

	case OFFLINE_VIDEO:

		 // For first frame openCV writer will be created
		 if (!writer.isOpened()){
		     writer.open(videoFilePath,vCodec,vFPS,frame.size(),vIsColored);
		 }
		 writer << frame;
  		 break;

	case FRAMES_TEXT_LIST:

		 // Create name for a the file which will store the current frame and write it
		 imageName.assign(videoFilePath,0,videoFilePath.length()-4);
		 char index[11];
 		 sprintf_s(index,"_%0.5d.jpg",frameNumber);
		 imageName += index;
		 //cvSaveImage(imageName.c_str(),frame);
		 imwrite(imageName,frame);
	
		 // Update the txt file
		 imageName.assign(videoFilePath,dirEnd+1,videoFilePath.length()-dirEnd-5);
		 imageName += index;
		 imageName += "\n";
		 fprintf_s(textFile,imageName.c_str());
  	     break;

 	 case SINGLE_IMAGE:	
		 //cvSaveImage(videoFilePath.c_str(),frame);
		 imwrite(videoFilePath,frame);
		 break;

	case UNKNOWN:
	case DEPLETED:
		 // Do nothing.
		 break;

	} //End switch

    frameNumber++;
	return;
}

/****************************************************************************/
// Return true if current source is video and not sequence of images.
bool cvExtFrameWriter::isVideoStreamWriting(void){
	return (framesDest == OFFLINE_VIDEO);
}

/****************************************************************************/
/***************************** cv:Mat Operations ****************************/
/****************************************************************************/

// Apply transformation T to column vectors v. V is not in homogenous coordinates
void cvExtTransform(const Mat& T, Mat& v){

	// Convert v to homogenous coordinates
	Mat Hom = Mat(v.rows+1,v.cols,CV_64FC(1),Scalar(1)); 
	v(Range(0,2),Range::all()).copyTo(Hom(Range(0,2),Range::all()));

	// Apply transform
	Hom = T*Hom;

	// Convert back to v	
	Hom(Range(0,2),Range::all()).copyTo(v);
	v.row(0) /= Hom.row(2);
	v.row(1) /= Hom.row(2);
}

/****************************************************************************/
// Check that 3x3 homography T is a valid transformation.
// Checks inner values of T, and that T transforms all pixels in image of srcSize inside dstSize
bool cvExtCheckTransformValid(const Mat& T, Size srcSize, Size dstSize){

	// Check the shape of the matrix
	if (T.empty())
	   return false;
	if (T.rows != 3)
	   return false;
	if (T.cols != 3)
	   return false;

	// Check for linear dependency of rows.
	Mat row1, row2;
	T(Range(0,1),Range(0,2)).copyTo(row1);
	T(Range(1,2),Range(0,2)).copyTo(row2);
	//T.row(0).copyTo(row1);
	//T.row(1).copyTo(row2);

	double epsilon = 0.000001; // in pixel units.
	row1 += epsilon;
	row2 += epsilon;
	row1 /= row2;
	Scalar mean;
	Scalar stddev;
	meanStdDev(row1,mean,stddev);
	double validVal = stddev[0]/(abs(mean[0])+epsilon);
    printf("Transformation validation: %g\n",validVal);
	if (validVal < 0.7)
	   return false;

	// Check that transformed source coordinates are inside destination coordinates.
	if ((srcSize.width <= 0)||(srcSize.height <= 0)){
	   // No need to check coordinates.
	   return true;	   
	}

	// Build coordinates.
	Mat footPrint;
	cvExtBuildFootPrintColsVector(srcSize,footPrint);
	cvExtTransform(T,footPrint);
	Rect_<double> r(0,0,dstSize.width,dstSize.height); 
	bool OK = true;
	for (int i=0;(i<footPrint.cols)&&OK;i++){
		Point tP = Point((int)footPrint.at<double>(0,i),(int)footPrint.at<double>(1,i));
		OK = (r.x <= tP.x && tP.x <= r.x + r.width) && (r.y <= tP.y && tP.y <= r.y + r.height);
	}
	
	if (!OK){
       printf("Transformation is out of borders\n");
	   return false;
	}

	return true;	
}


/****************************************************************************/
// Build vector (x,y) for each image corner and center. Store the reult is 2x5 matrix.
void cvExtBuildFootPrintColsVector(const Size s, Mat& cornersVec){
	Mat tmp = (Mat_<double>(5,2) << 0,0,s.width,0,0,s.height,s.width,s.height,s.width/2.0,s.height/2.0); 
	cornersVec = tmp.t();
}

// Build vector (x,y) for each image corner. Store the reult is 4x2 matrix.
void cvExtBuildCornersRowsVector(const Size s, Mat& cornersVec){
	cornersVec = (Mat_<double>(4,2) << 0,0,s.width,0,0,s.height,s.width,s.height); 
}

/****************************************************************************/
// Calculate normalized location of the projector that created the given footprint.
// Uses the perspective information in the foorprint
void cvExtFootPrintSourceLocation(Mat& footPrint, float& Xindex, float& Yindex){

	// Extract footprint center .
	float centX = (float)footPrint.at<double>(0,4);
	float centY = (float)footPrint.at<double>(1,4);

	// Calculate corner vectors
	footPrint(Range(0,1),Range::all()) -= centX;
	footPrint(Range(1,2),Range::all()) -= centY;

	// Take their inverse average.
	Xindex = (float)(-sum(footPrint.row(0)/4.0)[0]);
	Yindex = (float)(-sum(footPrint.row(1)/4.0)[0]);

	// TO do: Change the above calculation. It's results are changing depending on the zoom...
	return;
}

/****************************************************************************/
// Returns 3x3 perspective transformation for the corresponding 4 point pairs, 
// stored as row vectors in 4x2 matrix.
Mat cvExtgetPerspectiveTransform(const Mat& SrcRowVec, const Mat& DstRowVec){
	Point2f src[4];
	Point2f dst[4];
	cvtCornersMat_Point2fArray(SrcRowVec,src);
	cvtCornersMat_Point2fArray(DstRowVec,dst);    
	Mat P = getPerspectiveTransform(src,dst); 
	return P;
}


/****************************************************************************/
// Serialize matrix to char buffer. When writing buffer is allocated and must be freed by user
int serializeMat(Mat& M, char* &buf, char serializeFlags){

	// Determine which serialization to perform:
	int writeObj = serializeFlags&CVEXT_SERIALIZE_WRITE;
	int useFile  = serializeFlags&CVEXT_SERIALIZE_FILE;

	// ----------------------------------  Output object
	if (writeObj){             

		// If empty matrix, dont do anything.
		if (M.empty()){
		   return 0;
		}

		// Calculate header
		int cols = M.cols;
		int rows = M.rows;
		int chan = M.channels();
		int eSiz = (int)((M.dataend-M.datastart)/(cols*rows*chan));

		// Estimate total size of the data
		int totalDataSize = sizeof(cols)+sizeof(rows)+sizeof(chan)+sizeof(eSiz)+cols*rows*chan*eSiz;

		// Perform writing
		if (useFile){

			// Write to file
			ofstream out(buf, ios::out|ios::binary);			
			if (!out)
			   return 0;

			// Write header
			out.write((char*)&cols,sizeof(cols));
			out.write((char*)&rows,sizeof(rows));
			out.write((char*)&chan,sizeof(chan));
			out.write((char*)&eSiz,sizeof(eSiz));

			// Write data.
			if (M.isContinuous()){
		       out.write((char *)M.data,cols*rows*chan*eSiz);
			   out.close();
   			   return totalDataSize;
			}
			else{
			   out.close();
		       return 0;
		    }
		}
		else{

			// Allocate buffer
			buf = (char *)malloc(totalDataSize);
			if (!buf)
			   return 0;		

			// Define variable that will show which part of matrix we are serializing.
			int i = 0;

			// Write header
			memcpy(buf+i,(char*)&cols,sizeof(cols));   i += sizeof(cols);
			memcpy(buf+i,(char*)&rows,sizeof(rows));   i += sizeof(rows);
			memcpy(buf+i,(char*)&chan,sizeof(chan));   i += sizeof(chan);
			memcpy(buf+i,(char*)&eSiz,sizeof(eSiz));   i += sizeof(eSiz);

			// Write data.
			if (M.isContinuous()){
		       memcpy(buf+i,(char*)M.data,cols*rows*chan*eSiz);
   			   return totalDataSize;
			}
			else{
		       return 0;
		    }
		}


	}
	// ----------------------------------  Input object
	else {					

		// Define header variables
		int cols;
		int rows;
		int chan;
		int eSiz;

		if (useFile){

		   ifstream in(buf, ios::in|ios::binary);	
		   if (!in){
		      M = NULL_MATRIX;
			  return 0;
		   }

		   // Read header
		   in.read((char*)&cols,sizeof(cols));
		   in.read((char*)&rows,sizeof(rows));
		   in.read((char*)&chan,sizeof(chan));
		   in.read((char*)&eSiz,sizeof(eSiz));

		   // Estimate total size of the data
		   int totalDataSize = sizeof(cols)+sizeof(rows)+sizeof(chan)+sizeof(eSiz)+cols*rows*chan*eSiz;

		   // Determine type of the matrix 
		   int type = CV_BUILD_MATRIX_TYPE(eSiz,chan);

		   // Alocate Matrix.
		   M = Mat(rows,cols,type,Scalar(1));	
	
		   // Read data.
		   if (M.isContinuous()){
			  in.read((char *)M.data,cols*rows*chan*eSiz);
			  in.close();
			  return totalDataSize;
		   }
		   else{
		      in.close();
			  return 0;
		   }
		}
		else{
    
		   if (!buf){
			  M = NULL_MATRIX;
			  return 0;
		   }

		   // Define variable that will show which part of matrix we are serializing.
		   int i = 0;

		   // Read header
		   memcpy((char*)&cols,buf+i,sizeof(cols));   i += sizeof(cols);
		   memcpy((char*)&rows,buf+i,sizeof(rows));   i += sizeof(rows);
		   memcpy((char*)&chan,buf+i,sizeof(chan));   i += sizeof(chan);
		   memcpy((char*)&eSiz,buf+i,sizeof(eSiz));   i += sizeof(eSiz);

		   // Estimate total size of the data
		   int totalDataSize = sizeof(cols)+sizeof(rows)+sizeof(chan)+sizeof(eSiz)+cols*rows*chan*eSiz;

		   // Determine type of the matrix 
		   int type = CV_BUILD_MATRIX_TYPE(eSiz,chan);

		   // Alocate Matrix.
		   M = Mat(rows,cols,type,Scalar(1));	
		
		   // Read data.
		   if (M.isContinuous()){
		      memcpy((char*)M.data,buf+i,cols*rows*chan*eSiz);
    	      return totalDataSize;
		   }
		   else{
		      return 0;
		   }
		} // If use file
	} // If write obj
}

/****************************************************************************/
int saveMat(char* filename, Mat& M){
	return serializeMat(M,filename,CVEXT_SERIALIZE_FILE|CVEXT_SERIALIZE_WRITE);
}
int readMat(char* filename, Mat& M){
	return serializeMat(M,filename,CVEXT_SERIALIZE_FILE|CVEXT_SERIALIZE_READ);
}

/****************************************************************************/
// Serialize char buffer to file.
int serializeCharBuf(char* &buf, char* filePath, char serializeFlags, int bufSize){
	// Determine which serialization to perform:
	int writeObj = serializeFlags&CVEXT_SERIALIZE_WRITE;
	if (writeObj){             
	   // Write character buffer to file
	   ofstream out;
	   out.open(filePath, ios::out|ios::binary );
	   if (!out){
	      return 0;
	   }
	   out.write(buf,bufSize);	   
	   return bufSize;
	}
	else{
	   // Read character buffer from file
	   int fSize;
	   ifstream is;
	   is.open(filePath, ios::in|ios::binary );
	   if (!is){
	      return 0;
	   }
	   
	   // Calculate length of file:
	   is.seekg (0, ios::end);
	   fSize = (int)(is.tellg());
	   is.seekg (0, ios::beg);

	   // allocate enough memory.
	   buf = (char*)malloc(fSize);
	   if (!buf)
	      return 0;	 	
	   is.read(buf,fSize);
	   is.close();
	   return fSize;
	}
}

/****************************************************************************/
// Embedd char buffer in cv matrix or extract buffer from it. 
int embeddCharDataInMat(Mat& M, char* buf, int bufSize){
	if ((!buf)||(!bufSize))
	   return 0;
	M = Mat(Size(1,bufSize),CV_8UC1,buf);
	return M.empty() ? 0 : bufSize;
}

int extractCharDataFromMat(Mat& M, char* &buf, int &bufSize){
	if (M.empty())
  	   return 0;
	buf = (char*)M.datastart;
	bufSize = (int)(M.dataend - M.datastart);
	return bufSize;
}

/****************************************************************************/
void printMat(const Mat& M){
	switch ( (M.dataend-M.datastart) / (M.cols*M.rows*M.channels())){
	
	case sizeof(char):
		 printMatTemplate<unsigned char>(M,true);
  		 break;
	case sizeof(__int16):
		 printMatTemplate<__int16>(M,true);
  		 break;
	case sizeof(float):
		 printMatTemplate<float>(M,false);
  		 break;
	case sizeof(double):
		 printMatTemplate<double>(M,false);
  		 break;
	}
}
void deb(const Mat& M){
	imshow("Debug",M);
	cvWaitKey(0);
}

/****************************************************************************/
// Converts 4x2 double matrix into array of 4 Point2f. 
// 'dst' must be preallocated.
void cvtCornersMat_Point2fArray(const Mat& M, Point2f *dst){
	if (!dst)
	   return;

	for (int i=0;i<4;i++){
		dst[i] = Point2f((float)M.at<double>(i,0),(float)M.at<double>(i,1));
	}
}
	
/****************************************************************************/
// EOF.

