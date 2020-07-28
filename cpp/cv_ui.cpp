#include "cv_ui.h"
#include <iostream>

using namespace std;

CvUI::CvUI()
{
    mode = INIT;
    tl = Point(-1, -1);
    br = Point(-1, -1);
    newInit = false;
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

Point CvUI::get_tl()
{
    if(tl.x < br.x){
        return tl;
    }
    else{
        return br;
    }
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