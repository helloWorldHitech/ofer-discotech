#!/usr/bin/env python

'''
MOSSE tracking sample

This sample implements correlation-based tracking approach, described in [1].

Usage:
  mosse.py [--pause] [--no_keyboard] [--show_images] [<video source>]

  --pause  -  Start with playback paused at the first video frame.
              Useful for tracking target selection.

  --no_keyboard - calibration is not via keyboard clicks but via vocal instructions

  --show_images - in --no_keyboard mode there's no need to show images or interact with them.
                  this will make them show anyway 

  Draw rectangles around objects with a mouse to track them.

Keys:
  SPACE    - pause video
  c        - clear targets

[1] David S. Bolme et al. "Visual Object Tracking using Adaptive Correlation Filters"
    http://www.cs.colostate.edu/~bolme/publications/Bolme2010Tracking.pdf
'''

import numpy as np
import cv2
from common import draw_str, RectSelector
import video
import subprocess
import time
from httplib2 import Http
from mutagen.mp3 import MP3
import sys
import eyeControlConfig as ECC


def printDebug(message):
    if not ECC.muteDebugPrints:
        sys.stderr.write("Debug: "+message+'\n')



http = Http()

HIST_SIZE=10
STATE_HIST_SIZE = 200

def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

def make_gray_and_small(frame, but_not_gray=False):
    gray = frame if but_not_gray else cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    ratio = ECC.imageSizeRatio
    small = cv2.resize(gray,(int(round(w*ratio)),int(round(h*ratio))))
    return small


def play_async(filename,dir=ECC.mp3Dir):
    fullpath = "%s/%s"%(dir,filename) if (dir is not None and dir != "") else filename
    printDebug( "play: "+fullpath)
    player = subprocess.Popen(["mplayer", fullpath], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def mp3length(filename,dir=ECC.mp3Dir):
    fullpath = "%s/%s"%(dir,filename) if (dir is not None and dir != "") else filename
    return MP3(fullpath).info.length
    

class CALIBRATOR:
    def __init__(self):
        self.intro_length = mp3length(ECC.mp3FileMap['calibration introduction'])
        self.announcement_lengths = {}
        for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
            self.announcement_lengths[name] = self.announcement_length(name)

        
        self.capture_time_after_qued = mp3length(ECC.mp3FileMap['que'])+ ECC.captureTimeDelay

        self.calibrated_states = {}
        for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
            self.calibrated_states[name] = False

        self.calibrated = False
        self.started = False
        self.intro_started_at = False
        self.intro_ended = False
        self.now_calibrating = None #which state is being captured
        self.started_announcement_at = None
        self.finished_announcement = None
        self.qued_at = None

    def update(self,img):
        t= time.time()
        ret_val = None
        if self.calibrated:
            return ret_val

        if not self.started:
            self.started = True
            self.play_intro()
            self.intro_started_at = t
        else:
            if t-self.intro_started_at >= self.intro_length:
                self.intro_ended = True
                
            if self.intro_ended:
                if self.now_calibrating is None:
                    self.now_calibrating = self.next_state_to_calibrate()
                    self.play_announcement(self.now_calibrating)
                    self.started_announcement_at = t
                else:
                    if t-self.started_announcement_at >= self.announcement_length(self.now_calibrating):
                        self.finished_announcement = True

                    if self.finished_announcement:
                        if self.qued_at is None:
                            self.qued_at = t
                            self.play_que()
                        else:
                            if t-self.qued_at >= self.capture_time_after_qued:
                                self.calibrated_states[self.now_calibrating] = True
                                if self.all_calibrated():
                                    self.calibrated = True
                                    self.play_final_ack()
                                else:
                                    self.play_ack()

                                ret_val = [self.now_calibrating, self.calibrated,img]

                                self.now_calibrating = None
                                self.qued_at = None
                                self.finished_announcement = False
        return ret_val
                    

    def play_intro(self):
        play_async(ECC.mp3FileMap['calibration introduction'])

    def play_que(self):
        play_async(ECC.mp3FileMap['que'])

    def play_ack(self):
        play_async(ECC.mp3FileMap['ack'])
        
    def play_final_ack(self):
        play_async(ECC.mp3FileMap['final ack'])

    def play_announcement(self,state):
        play_async(self.announcement_file(state))

    def announcement_file(self,state):
        return ECC.mp3FileMap['instruction'][state]




    def all_calibrated(self):
        for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
            if not self.calibrated_states[name]:
                return False
        return True

    def next_state_to_calibrate(self):
        for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
            if not self.calibrated_states[name]:
                return name

    def announcement_length(self,state):
        return mp3length(self.announcement_file(state))

                         

class STATE:
    def __init__(self):
        self.baseTime = time.time()
        self.frameNumber = -1
        self.frameTime = None
        self.psrs = None
        self.raw_choice = None
        self.in_state = ECC.defaultState
        self.in_state_since = -1
        self.exiting = False
        self.exiting_since = None
        self.to_state = None
        self.to_state_since = None
        self.broadcasted = False

    def __str__(self):
        if self.exiting:
            return "%04d:%3.2f : %7s %7s(%3.2f) [%3s]-> (%3.2f) -> %7s(%3.2f)"%(self.frameNumber,self.frameTime,self.raw_choice,self.in_state,self.in_state_since,self.broadcasted,self.exiting_since,self.to_state,self.to_state_since)
        else:
            return "%04d:%3.2f : %7s %7s(%3.2f) [%3s]"%(self.frameNumber,self.frameTime,self.raw_choice,self.in_state,self.in_state_since,self.broadcasted)
                                                                           
    def update(self,psrs):
        self.frameTime = time.time()-self.baseTime
        self.frameNumber = self.frameNumber + 1
        self.psrs = psrs
        
        # make a raw choice
        maxPsr = None
        chosen = None
        for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
            if maxPsr is None or psrs[name]>maxPsr:
                maxPsr = psrs[name]
                chosen = name

        if chosen is None:
            self.raw_choice = ECC.defaultState
        elif chosen == ECC.referenceState:
            self.raw_choice = chosen
        elif psrs[chosen] > psrs[ECC.referenceState] * ECC.psrRefRatio:
            self.raw_choice = chosen
        else:
            self.raw_choice = ECC.defaultState
        
        # update state
        
        if self.raw_choice == self.in_state:
            # current state is reinforced
            self.exiting = False
            self.exiting_since = None
            self.to_state = None
            self.to_state_since = None
        elif self.raw_choice != self.to_state:
            # update the exiting state
            if not self.exiting:
                self.exiting = True
                self.exiting_since = self.frameTime
            
            self.to_state = self.raw_choice
            self.to_state_since = self.frameTime
        else:
            pass
            # still exiting to the to_state. nothing to update

        # check if should move into the to_state
        if self.exiting:
            timeToIn, timeToOut, timeToBroadcast, broadcast, defaultOut = ECC.stateKeyAndProperties[self.to_state]
            if (self.frameTime - self.to_state_since) >= timeToIn:
                # the to state becomes the state
                self.in_state = self.to_state
                self.in_state_since = self.to_state_since
                self.exiting = False
                self.exiting_since = None
                self.to_state = None
                self.to_state_since = None
                self.broadcasted = False

        # check if should move out of the in_state
        timeToIn, timeToOut, timeToBroadcast, broadcast, defaultOut = ECC.stateKeyAndProperties[self.in_state]
        if self.exiting and (self.frameTime - self.exiting_since) >= timeToOut:
            self.in_state = ECC.defaultState
            self.in_state_since = self.exiting_since
            self.exiting = False
            self.exiting_since = None
            self.to_state = None
            self.to_state_since = None
            self.broadcasted = False           

        # check if need to broadcast
        toBroadcast = []
        timeToIn, timeToOut, timeToBroadcast, broadcast, defaultOut = ECC.stateKeyAndProperties[self.in_state]
        if not self.broadcasted and (self.frameTime - self.in_state_since) >= timeToBroadcast:
            if broadcast is not None:
                toBroadcast.append(broadcast)
            self.broadcasted = True

        return toBroadcast

class MOSSE:
    def __init__(self, frame, rect,toStdOut,toSound,to2048):
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        self.posHist = []
        self.std = 999

        self.positionImages = {}
        self.positionFilters = {}
        self.positionBetterFilters = {}
        self.positionsInitialized = False


        self.toStdOut = toStdOut
        self.toSound = toSound
        self.to2048 = to2048
        
        self.state = STATE()

        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self.H1, self.H2, self.G = self.createFilter(img)

        self.update_kernel()
        self.update(frame)

    def createFilter(self,img,gaussianSize = 2.0, numMoves = ECC.numMoves):
        h, w = img.shape
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), gaussianSize)
        g /= g.max()
        G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = np.zeros_like(G)
        H2 = np.zeros_like(G)
        for i in xrange(numMoves):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            H1 += cv2.mulSpectrums(G, A, 0, conjB=True)
            H2 += cv2.mulSpectrums(A, A, 0, conjB=True)
    
        return H1, H2, G

    def haveAllPositions(self):
        for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
            if name not in self.positionImages:
                return False
        return True

    def initializeFilters(self):
        self.positionsInitialized = True
        commonH2 = None
        for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
            H1, H2, G = self.positionFilters[name]
            if commonH2 is None:
                commonH2 = H2.copy()
            else:
                commonH2 += H2
        for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
            H1, H2, G = self.positionFilters[name]
            self.positionBetterFilters[name] = divSpec(H1, commonH2)
            self.positionBetterFilters[name][...,1] *= -1



    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            self.posHist=[]
            return
        self.pos = x+dx, y+dy
        
        self.posHist.append(self.pos)


        if len(self.posHist) > HIST_SIZE:
            self.posHist.pop(0);

            #std
            an=np.array(self.posHist)
            s=np.std(an, axis=0)
            self.std=s[0]+s[1]
        
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def savePosition(self, which):
        printDebug( 'save %s'%which)
        self.positionImages[which] = self.last_img
        self.positionImages[which] = self.preprocess(self.positionImages[which])
        self.positionFilters[which] = self.createFilter(self.last_img)
        im = self.positionImages[which]
        #cv2.imshow(which,np.uint8((im-im.min())/im.ptp()*255))
        self.positionsInitialized = False

    def send(self, message):
        if self.to2048:
            resp, content = http.request("http://localhost:8889/a/message/new", "POST", message)
        if self.toStdOut:
            printDebug( "Sending: "+message)
            print message
	    sys.stdout.flush();

        if self.toSound:
            play_async(ECC.mp3FileMap['recognition'][message])

            #player = subprocess.Popen(["mplayer", "mp3/%s.mp3"%message.replace("*","")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
            
        
        
    def draw_state(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            for pos in self.posHist:
                (xx, yy) = pos
                pt = (int(xx), int(yy))
                cv2.circle(vis, pt, 2, (0, 0, 255), -1)

#            cv2.putText(vis, '%.2f' % self.std, pt, cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            if self.std < 0.5:
                cv2.putText(vis, 'FIX', (pt[0], pt[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

            if self.haveAllPositions() and not self.positionsInitialized:
                self.initializeFilters()
                
            if self.positionsInitialized:
                psrs = {}
                newPsrs = {}
                pimg = self.preprocess(self.last_img)
                for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
                    resp, (ddx, ddy), psrs[name] = self.correlate(self.positionImages[name])
                    resp, (ddx, ddy), newPsrs[name] = self.correlate_(self.positionBetterFilters[name],pimg)

                    cv2.putText(vis, '%s: %.2f' % (letter, psrs[name]), (pt[0], pt[1]+pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                    cv2.putText(vis, '%s: %.2f' % (letter, newPsrs[name]), (pt[0]+225, pt[1]+pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

                toBroadcast =  self.state.update(newPsrs)
                printDebug( str(self.state))


                cv2.putText(vis, self.state.raw_choice, (pt[0]+0, pt[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (2,2,255), 4)
                cv2.putText(vis, 'B' if self.state.broadcasted else '-', (pt[0]-25, pt[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (2,2,255), 4)
                cv2.putText(vis, self.state.in_state, (pt[0]+0, pt[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (2,2,255), 4)
            
            
                for b in toBroadcast:
                    self.send(b)

        else:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        return self.correlate_(self.H, img)

    #(static)
    def correlate_(self, H, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1



class App:
    def __init__(self, video_src, no_keyboard,show_images,toStd,toSound,to2048,paused = False):
        self.toStd = toStd
        self.toSound = toSound
        self.to2048 = to2048
        self.no_keyboard = no_keyboard
        self.show_images = show_images
        self.cap = video.create_capture(video_src)
        #first few frames tend to arrive shifted; skip.
        for i in xrange(5):
            self.cap.read()
        _, self.frame = self.cap.read()
        smallgray = make_gray_and_small(self.frame)
        h, w = smallgray.shape[:2]
        if(no_keyboard is False):
            cv2.imshow('frame', self.frame)
            self.rect_sel = RectSelector('frame', self.onrect)
        iwp=ECC.initialWindowPortion
        self.trackers = [ MOSSE(smallgray,
                                [int(w*(1.-iwp)/2.),int(h*(1.-iwp)/2.),int(w*(1.+iwp)/2.),int(h*(1.+iwp)/2.)],
                                self.toStd,self.toSound,self.to2048)]

        self.calibrator = CALIBRATOR()
        self.paused = paused

    def onrect(self, rect):
        frame_gray = make_gray_and_small(self.frame)
        tracker = MOSSE(frame_gray, rect,self.toStd,self.toSound,self.to2048)
        self.trackers.append(tracker)
                         

    def run_no_keyboard(self,show_images):
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            frame_gray = make_gray_and_small(self.frame)
            for tracker in self.trackers:
                tracker.update(frame_gray)
            res = self.calibrator.update(frame_gray)
            if res is not None: 
                captured_state, calibration_ended,_ = res
                if len(self.trackers) > 0:
                    self.trackers[-1].savePosition(captured_state)
                printDebug( "captured: "+ captured_state)
                if calibration_ended:
                    printDebug( "calibration ended.")

            cv2.waitKey(10)
            vis = make_gray_and_small(self.frame,but_not_gray=True).copy()
            for tracker in self.trackers:
                tracker.draw_state(vis)
            if show_images:
                if len(self.trackers) > 0:
                    cv2.imshow('tracker state', self.trackers[-1].state_vis)

                cv2.imshow('frame', vis)
            

    def run_from_keyboard(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                frame_gray = make_gray_and_small(self.frame)
                for tracker in self.trackers:
                    tracker.update(frame_gray)

            vis = make_gray_and_small(self.frame,but_not_gray=True).copy()
            for tracker in self.trackers:
                tracker.draw_state(vis)
            if len(self.trackers) > 0:
                cv2.imshow('tracker state', self.trackers[-1].state_vis)
            self.rect_sel.draw(vis)

            cv2.imshow('frame', vis)
            ch = cv2.waitKey(10)
            if ch == 27:
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []

            if len(self.trackers) > 0:
                for key, (name, letter, pos) in ECC.positionKeyAndName.iteritems():
                    if ch == ord(key):
                        self.trackers[-1].savePosition(name)

    def run(self):
        if self.no_keyboard:
            self.run_no_keyboard(self.show_images)
        else:
            self.run_from_keyboard()
if __name__ == '__main__':
    printDebug( __doc__)
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['pause', 'no_keyboard', 'show_images', 'voice_feedback'])

    opts = dict(opts)
    try: video_src = args[0]
    except: video_src = '0'
    no_keyboard = '--no_keyboard' in opts
    show_images = not no_keyboard or '--show_images' in opts
    paused = '--pause' in opts
    toStdOut = True
    to2048 = False
    toSound = '--voice_feedback' in opts

    App(video_src, no_keyboard,show_images, toStdOut,toSound,to2048,paused).run()
