/****************************************************************************/
// Converts image to drawing
#include "cv_extension.h"
#include "stdio.h"

/****************************************************************************/

// Names of windows
#define winNameO "Remote Control"
#define winNameP "Comics Image"


// Pointers to auxilary images
static IplImage* srcImageOrig  = NULL;
static IplImage* dstImageOrig  = NULL;
static IplImage* srcImage      = NULL;
static IplImage* dstImage      = NULL;

// Arguments to image2comics converter
static int _intensNormal	= 9;
static int _looseDitails	= 2;
static int _edgeSharpness	= 8;
static int _colorNormal		= 15;
static int _colorBits		= 3;
static int _fast 	   	    = 1;

/****************************************************************************/
// Call back function: Applied to each image stored in srcImage and returns result in dstImage
void callBackFun(int arg){
	cvExtPhoto2Pencil(srcImage,dstImage,_intensNormal/10.,_looseDitails,
		                                _edgeSharpness,
										_colorNormal,_colorBits,_fast);
	cvShowImage(winNameP,dstImage );
}

/****************************************************************************/
// Main method for converting media to comics
int main(int argc, char** argv ){

	cvExtFrameGrabber*	media   = NULL;	   // Media source 
	cvExtFrameWriter*   vWriter = NULL;	   // Media destination 
	int keyPressed;                        // Current keyboard command

	// ---------------------------------------
	// Initialize media source
	
	string resultPath;

	if (argc < 2){
	   fprintf(stderr,"1. Connecting to online camera...\n");
	   media      = new cvExtFrameGrabber();
	   resultPath = "tmp_com.avi";
	}
	else{
	   fprintf(stderr,"1. Loading media: %s\n",argv[1]);
       media      = new cvExtFrameGrabber(argv[1]);
	
	   resultPath = argv[1];
	   resultPath.insert(resultPath.length()-4,"_com");
	}

	// Check if media source is working
	srcImageOrig = media->grabFrameIpl();
	if (!srcImageOrig){
	   fprintf(stderr,"2. Error: Unable to acquire media!\n");
	   delete media;
	   cvWaitKey(0);
	   return -1;
	}

	// Open media writer
	vWriter = new cvExtFrameWriter(resultPath.c_str(),media->getFPS());
    fprintf(stderr,"2. Output media : %s\n",resultPath.c_str());

	// Open remote control window
	cvNamedWindow(winNameO, 1);
	cvResizeWindow(winNameO, 285, 300 );
	cvCreateTrackbar("Brightness" ,winNameO, &_intensNormal  , 10, callBackFun);
	cvCreateTrackbar("Ditails   " ,winNameO, &_looseDitails  , 4,  callBackFun);
	cvCreateTrackbar("Bold Strok" ,winNameO, &_edgeSharpness , 20, callBackFun);
	cvCreateTrackbar("Color     " ,winNameO, &_colorNormal   , 30, callBackFun);
	cvCreateTrackbar("Col Ver.  " ,winNameO, &_colorBits     , 8 , callBackFun);
	cvCreateTrackbar("Upgrade   " ,winNameO, &_fast          , 1, callBackFun);
	
	// Open Image window
    cvNamedWindow(winNameP, 1);

	// Images will be resized so their longest edge will be maximum 800 pixels maximum.
	// The reason is letting each image to fit the screen
	CvSize curSize = cvGetSize(srcImageOrig);
	int longest    = CVEXT_MAX(curSize.height,curSize.width);
	double ratio   = 800.0/longest;
	       ratio   = (ratio>1)?1:ratio;
	CvSize newSize = cvSize(cvRound(ratio*curSize.width),cvRound(ratio*curSize.height));
	fprintf(stderr,"3. Media is displayed in %dx%d resolution\n\n",newSize.height,newSize.width);
	srcImage       = cvCreateImage( newSize, srcImageOrig->depth, srcImageOrig->nChannels);

	// Allocate memory for destination image
	dstImageOrig   = cvCreateImage( curSize, IPL_DEPTH_8U       , srcImageOrig->nChannels );
	dstImage       = cvCreateImage( newSize, IPL_DEPTH_8U       , srcImageOrig->nChannels);

	// ---------------------------------------
	// processing loop. 

	bool stepByStep = false; // If true media will require a keystroke before continuing to the next frame

	while (srcImageOrig){

		fprintf(stderr,"Processing frame number %0.5d",media->getCurFrameNumber());
		
		// Convert current frame to comics and display it
		cvResize(srcImageOrig,srcImage,CV_INTER_LINEAR);      
        callBackFun(0);

	    // Get the next frame.
		cvReleaseImage(&srcImageOrig);
		srcImageOrig = media->grabFrameIpl();

		// Let user enter his command (Pause automatically on first frame to let him edit the image)
		if ((media->getCurFrameNumber() == 2)||(stepByStep)||(!srcImageOrig)){
		   fprintf(stderr," - Press enter to continue\n");
   		   keyPressed = cvWaitKey(0);
		}
		else{
		   fprintf(stderr,"\n");
   		   keyPressed = cvWaitKey(1);
		}

	    // Write result to disk
	    cvResize(dstImage,dstImageOrig,CV_INTER_LINEAR);
		vWriter->writeFrame(dstImageOrig);

		// Proccess user commands (Pause automatically on first frame)
 		switch (keyPressed){
		case CVEXT_NO_KEY:                 			 
				break;

		case CVEXT_QUIT: 
	  			// Quit
				cvExtReleaseImageCheckNULL(&srcImageOrig);
				break;

		case CVEXT_SBS_MODE:
			    stepByStep = !stepByStep;
			
		case CVEXT_PAUSE: 
			    cvWaitKey(0);
				break;

		case CVEXT_MOVE_WIN: 
				cvMoveWindow(winNameO, 30, 30 );
				cvMoveWindow(winNameP, 50, 50 );
				break;

		}
	}
	// ---------------------------------------
	

	// Close program
	cvDestroyWindow(winNameO);
	cvDestroyWindow(winNameP);

	// Release auxilarry images
	cvExtReleaseImageCheckNULL(&dstImageOrig);
    cvExtReleaseImageCheckNULL(&srcImage);
    cvExtReleaseImageCheckNULL(&dstImage);

	// Release media objects
    delete media;
    delete vWriter;

	return 0;
}

/****************************************************************************/
// EOF.
