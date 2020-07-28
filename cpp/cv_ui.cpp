#include "cv_ui.h"
#include <iostream>

using namespace std;
using namespace cv;

MODE CvUI::mode = INIT;
Point CvUI::tl = Point(-1, -1);
Point CvUI::br = Point(-1, -1);
bool CvUI::newInit = false;

CvUI::CvUI()
{
}

CvUI::~CvUI()
{
}

void CvUI::OnMouse(int event, int x, int y, int flags, void* ustc)
{
    if(event == EVENT_LBUTTONDOWN && mode == INIT){
        tl = Point(x, y);
        br = Point(x, y);
        mode = SELECT;
    }
    else if(event == EVENT_MOUSEMOVE && mode == SELECT){
        br = Point(x, y);
    }
    else if(event == EVENT_LBUTTONDOWN && mode == SELECT){
        br = Point(x, y);
        mode = INIT;
        newInit = true; 
    }
}

Point CvUI::GetTl()
{
    if(tl.x < br.x){
        return tl;
    }
    else{
        return br;
    }
}

Point CvUI::GetBr()
{
    if(tl.x < br.x){
        return br; 
    }
    else{
        return tl;
    }
}

Rect2d CvUI::GetBb()
{
    Point tl_ = GetTl();
    Point br_ = GetBr();
    Rect2d bb;
    bb.x = min(tl_.x, br_.x);
    bb.y = min(tl_.y, br_.y);
    bb.width = abs(br_.x - tl_.x);
    bb.height = abs(br_.y - tl_.y);
    return bb;
}

    // def get_tl(self):
    //     return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

    // def get_br(self):
    //     return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

    // def get_bb(self):
    //     tl = self.get_tl()
    //     br = self.get_br()

    //     bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
    //     return bb