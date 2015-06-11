# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:32:44 2014

@author: Hjalmar K. Turesson
"""
import os
import sys
from time import sleep, perf_counter
from glob import glob
import numpy as np
from threading import Event, Thread
from multiprocessing import Process
from datetime import datetime
import subprocess
#import subprocess as sp
import pygame
import pygame.camera
from pygame.locals import Rect
from pylibftdi import BitBangDevice
from bayercy import bayer
import flycapture2 as fc2
import pandas as pd
import tables
import scipy
from scipy.io import wavfile
import pyaudio
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
import imageio
import time
import os, psutil, resource
#import visvis as vv

# Setting GST_DEBUG_DUMP_DOT_DIR environment variable enables us to
# have a dotfile generated. The environment variable cannot be set inside the class.
os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "/tmp"
# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, Gtk
Gst.debug_set_active(True)
Gst.debug_set_default_threshold(2)
GObject.threads_init()
Gst.init(None)


def play_sound(x, fs, volume=0.5):
    """
    volume - 0 - 1.0, None if the sound should be played back at original volume.
    """
    tmp_fn = '/tmp/tmp_snd.wav'
        
    if not volume is None:
        volume = min(volume, 1.0) # Not above 1.0
        if type(x) is np.ndarray:
            if  x.dtype.kind is 'f':
                x = np.int16(0.5 * 2**16 * volume * x/x.max())
            elif x.dtype == 'int16':
                x = np.int16(volume * (2**16) * x/(2 * x.max()))
            elif x.dtype == 'int32':
                x = np.int32(volume * (2**32) * x/(2 * x.max()))

    # write the uncompressed wavfile
    wavfile.write(tmp_fn, fs, x)
    
    cmd = ['aplay',
           '-r %d' % (fs),
            tmp_fn]
    # Use sox to transcode wav to flac
    subprocess.call(cmd)
    # Remove wav file
    os.remove(tmp_fn)
    
def play_sound_pa(x, fs, volume=0.5, pa=None, normalize=True):
    """
    Plays a sound using PyAudio
    
    x       - The signal to be playes, a numpy.ndarray or an array_like that can be 
              converted to a numpy.ndarray.
    fs      - Sampling rate.
    volume  - Playback volume, between 0 and 1.
    pa      - An instance of pyaudio.PyAudio or
              None (in which case an instance of pyaudio.PyAudio will be created).

    Requires PyAudio
              
    By Hjalmar K. Turesson, 28-01-2015
    """
    
    close_pa = False
    if pa is None:
        pa = pyaudio.PyAudio()
        close_pa = True
    elif not type(pa) is pyaudio.PyAudio:
        print('p has to be type pyaudio.PyAudio or None')
        return 0
    
    # Check type of x
    if not type(x) is np.ndarray:
        try:
            x = np.array(x)
        except:
            if close_pa:
                pa.terminate()
            print('x should be a numpy.ndarray or an array_like object.')
            return 0

    # Check dimensions of x. 1 or 2 channels
    channels=1
    if x.ndim > 2:
        print('x cannot have more than two dimensions')
        return 0
    elif (x.ndim == 2):
        if x.shape[0] == 1 and x.shape[1] > 1:
            # Already 1-d, just need to be flattened
            x = x.flatten()
        elif x.shape[0] == 2 and x.shape[1] > 1:
            # 2 channel, just transpose
            x = x.T
            channels=2
          
    if normalize:
        # Normalize x
        maxval = ((2**16-1)-2**15)  # Play sound as int16
        x = x/x.max()
        sound = np.int16(maxval*x*volume)
    elif not x.dtype is np.dtype('int16'):
        sound = np.int16(x)
    else:
        sound = x

    #fs = int(p.get_default_output_device_info()['defaultSampleRate'])
    CHUNK = 2048
    stream = pa.open(format=pyaudio.paInt16,
                     channels=channels,
                     rate=int(fs),
                     output=True,
                     frames_per_buffer=CHUNK,
                     output_device_index=None)
    
    for ix in range(0, sound.shape[0], CHUNK):
        stream.write((sound[ix:ix+CHUNK]).tostring())
    stream.stop_stream()
    stream.close()
    if close_pa:
        # close PyAudio
        pa.terminate()


def record_experiment_Lina():    
    record_ATrain_experiment(1272)


def record_experiment_Sidarta():    
    record_ATrain_experiment(1303)    


def record_ATrain_experiment(animal_ID):
    """
    anima_ID - identifying number of the marmoset in the experiment.
               e.g.: 1272 (Lina), 1303 (Sidarta), 1086 (Suria) 
    
    By Hjalmar K. Turesson 2015-04-10
    """
    d = '/media/%s/VitaminD/Sound/ICe/ATrain/continous/sound/' % os.getlogin()

    date = datetime.now().strftime('%Y%m%d')
    time = datetime.now().strftime('%H-%M-%S')

    fn = '%sAT_%s_%s_01_%s_cont.wav' % (d , date, time, animal_ID)
    
    record_sound_USBPre2(file_name=fn,  duration=1020)
    stereo2mono(fn)


def record_sound_USBPre2(file_name=None, duration=60):
    """
    duration -- duration of recording in seconds.
    
    Alsa info:
    http://jan.newmarch.name/LinuxSound/Sampled/Alsa/
    
    By Hjalmar K. Turesson 2015-04-10
    """
    
    if file_name is None:
        file_name = 'rec_%s.wav' % datetime.now().strftime('%Y%m%d_%H-%M-%S')
    
    cmd = ['arecord',
           '--device=iec958:USBPre2,0',
           '--rate=192000',
           '--file-type=wav',
           '--format=S16_LE',
           '--duration=%d' % round(duration),
           '--channels=2',
           file_name]
    
    subprocess.call(cmd)


def stereo2mono(fname, append_mono_to_fname=True):
    
    fs, x = wavfile.read(fname)
    # select loudest channel
    ch = np.argmax(np.abs(x).sum(axis=0))
    fn_out = fname.replace('.wav','_mono.wav')
    wavfile.write(fn_out, fs, x[:,ch].flatten())
    if not append_mono_to_fname:
        os.remove(fn)
        os.rename(fn_out, fn)


def h5_frames_to_video_avconv(h5_fn, vid_fn=None, fps=30, ffopts=[]):
    """
    """
    if vid_fn is None:
        vid_fn = h5_fn.split('.')[0] + '.mp4'

    h5f = tables.open_file(h5_fn, mode='r')
    frames = h5f.root.recording.frames
    frames_ts = h5f.root.recording.timestamps
    #tsSpecs, ts_fmt = prep_render_timestamp_for_vid(frames[:,:,:3], frames_ts[0])
    
    ts_ix = 0
    p = None
    for fr_ix in range(0,frames.shape[2],3):
        fr = frames[:,:,fr_ix:fr_ix+3]
        ts = frames_ts[ts_ix]
     #   fr = render_timestamp(frames[:,:,fr_ix:fr_ix+3], ts, tsSpecs, ts_fmt)
        ts_ix += 1
        if p is None:
            cmd = ['avconv',
                   '-y', '-s',
                   '%dx%d' % (fr.shape[1], fr.shape[0]),
                   '-f', 'rawvideo',
                   '-pix_fmt', 'rgb24',
                   '-r', str(fps),
                   '-i', '-',
                   '-vcodec', 'mpeg4',
                   '-b', '3000k',
                   ] + ffopts + [vid_fn]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.stdin.write(fr.tostring())
    p.stdin.close()
    p.wait()    
    
    h5f.close()


def h5_frames_to_png(h5_fn, vid_fn):
    """
    """

    h5f = tables.open_file(h5_fn, mode='r')
    frames = h5f.root.recording.frames
    frames_ts = h5f.root.recording.timestamps
    #tsSpecs, ts_fmt = prep_render_timestamp_for_vid(frames[:,:,:3], frames_ts[0])
    d = '/home/hjalmar/data/video/Sideview/video1/Frames/'
    ts_ix = 0
    ts_fn = d+'timestamps.txt'
    f = open(ts_fn,'w')
    f.write('timestamps in seconds:/n')
    for fr_ix in range(0,frames.shape[2],3):
        fr = frames[:,:,fr_ix:fr_ix+3]
        ts = frames_ts[ts_ix]
     #   fr = render_timestamp(frames[:,:,fr_ix:fr_ix+3], ts, tsSpecs, ts_fmt)
        ts_ix += 1        
        scipy.misc.imsave('%ssuria_sideview_141228_frame_%d.png' % (d, ts_ix), fr)
        f.write(str(ts)+'\n')
    h5f.close()


def frames_to_video(frames, path, fps=15, ffopts=[]):
    # '-c:v', 'h264', '-f', 'h264',
    p = None
    for fr in frames:
        if p is None:
            cmd = ['avconv',
                   '-y', '-s',
                   '%dx%d' % (fr.shape[1], fr.shape[0]),
                   '-f', 'rawvideo',
                   '-pix_fmt', 'rgb24',
                   '-r', str(fps),
                   '-i', '-',
                   '-vcodec', 'mpeg4',
                   '-b', '3000k',
                   ] + ffopts + [path]
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.stdin.write(fr.tostring())
    p.stdin.close()
    p.wait()


def multiproc_save_frames(path, frames, frames_ts=None):
    """
    """
    if frames_ts is None:
        p = Process(target=save_frames, args=(path, frames))
    else:
        p = Process(target=save_frames_and_timestamps,
                    args=(path, frames, frames_ts))

    p.start()


def save_frames(path, frames):
    """
    """
    sleep(0.5)
    np.save(path, np.array(frames))


def save_frames_and_timestamps(path, frames, ts):
    """
    """
    sleep(0.5)
    np.savez(path, frames=np.array(frames), ts=np.array(ts))
    


class VideoCapture_Continous:
    """
    """
    def __init__(self, file_name, fps=30, frame_size=(640, 480),
                 dev='/dev/video0', display=False, duration=60,
                 start_time=None, overwrite_existing_file=False):
        """
        file_name               --
        fps                     -- frames per second
        frame_size              -- 
        dev                     --
        display                 --
        duration                -- In seconds
        start_time              -- datetime.datetime
        duration                -- Approximate duration of recording in minutes        
        overwrite_existing_file --
        
        Hjalmar K. Turesson, 2014-11-20
        """
        self.file_name = file_name
        self.file_name_initial = self.file_name
        self.frame_size = frame_size
        self.fps = fps
        self.duration = duration  # in minutes
        self.dev = dev
        self.n_frames = 0
        self.display = display        
        self.overwrite_existing_file = overwrite_existing_file
        self.start_time = start_time
        self.camera_running = False
        self.color_mode = 'RGB'
        self._frmatom = tables.UInt8Atom(shape=())
        self._frmshape = (frame_size[1], frame_size[0], 0)
        self._exprows = self.duration * self.fps
        #self._filters = tables.Filters(complevel=9, complib='blosc')
        self._tsatom = tables.Float64Atom(shape=())
        self._tsshape = (0,)

    def _run(self):
        """
        """
        self.cam.start()
        self.camera_running = True
        self.recording_time = 0.0
        if self.start_time is None:
            self.start_time = datetime.now()
        t0 = self.start_time.timestamp()
                        
        if self.display:
            # create a display surface.
            self.screen = pygame.display.set_mode((640,480))
            # create a surface to capture to.  for performance purposes
            # bit depth is the same as that of the display surface.
            self.image = pygame.surface.Surface(self.frame_size, 0, self.screen)
            # Create Pygame clock object.  
            clock = pygame.time.Clock()
        else:
            self.image = pygame.surface.Surface(self.frame_size, 0)
                
        while not self._event.isSet():
            self.ts = datetime.now().timestamp() - t0
            self.image = self.cam.get_image(self.image)
            self.frame = pygame.surfarray.array3d(self.image).transpose((1, 0, 2))
            self.frames.append(self.frame)
            self.n_frames += 1
            self.timestamps.append([self.ts])
            # Do not go faster than this framerate.
            milliseconds = clock.tick(self.fps)
            self.recording_time += milliseconds / 1000.0 
            
            if self.display:
                # blit it to the display surface.
                self.screen.blit(self.image, (0,0))
                # Print framerate and playtime in titlebar.
                text = "FPS: %1.2f   Recording time: %d min %1.2f s" % \
                        (clock.get_fps(), self.recording_time // 60,
                         self.recording_time % 60)
                pygame.display.set_caption(text)
                # Update Pygame display.                
                pygame.display.flip()
                
        self.cam.stop()
        self.camera_running = False
        self.recording_parameters.append([(self.color_mode,
                                           self.fps,
                                           self.start_time.strftime("%Y-%m-%d_%H:%M:%S.%f"),
                                           self.start_time.timestamp())])
        self.h5f.close()
        # Finish Pygame.  
        pygame.quit()        
        
    def start_recording(self):
        """
        """
        if self.camera_running:
            print('Recording is already running.\nTo restart, stop recording first.')
        else:
            if not self.overwrite_existing_file:
                ext = self.file_name.split('.')[-1]
                len_ext = len(ext)+1
                n = 1
                while self.file_name in glob(self.file_name):
                    self.file_name = '%s_%d.%s' % (self.file_name_initial[:-len_ext],
                                                   n, ext)
                    n += 1
    
            self.h5f = tables.open_file(self.file_name, mode='w')
            rec_grp = self.h5f.createGroup( self.h5f.root, 'recording' )
#            self.frames = self.h5f.create_earray(rec_grp, "frames",
#                                                 atom=self._frmatom,
#                                                 shape=self._frmshape,
#                                                 expectedrows=self._exprows,
#                                                 filters=self._filters)
            self.frames = self.h5f.create_earray(rec_grp, "frames",
                                                 atom=self._frmatom,
                                                 shape=self._frmshape,
                                                 expectedrows=self._exprows)            
#            self.timestamps = self.h5f.create_earray(rec_grp, "timestamps",
#                                                 atom=self._tsatom,
#                                                 shape=self._tsshape,
#                                                 expectedrows=self._exprows,
#                                                 filters=self._filters) 
            self.timestamps = self.h5f.create_earray(rec_grp, "timestamps",
                                                     atom=self._tsatom,
                                                     shape=self._tsshape,
                                                     expectedrows=self._exprows)                                                 

            # Table holding recording related data.
            class rec_params(tables.IsDescription):
                color_mode = tables.StringCol(3)  
                # frames per second
                fps = tables.Int64Col()
                # Recording start date and time as a string
                start_datetime = tables.StringCol(26)
                # Recording start time as time stamp
                start_time_ts = tables.Float64Col()

            self.recording_parameters = self.h5f.createTable(rec_grp,
                                                             'recording_params',
                                                             rec_params,
                                                             "Recording parameters")
                                                 
            pygame.init()
            pygame.camera.init()
            self.cam = pygame.camera.Camera(self.dev,
                                            self.frame_size,
                                            self.color_mode)
            self._event = Event()
            self._thread = Thread(target=self._run)
            self._thread.start()        
        
    def stop_recording(self):
        """
        """
        if self.camera_running:
            self._event.set()
        else:
            print("Recording was not running, thus not stopped.")
 

class VideoCapture_BufferControl:
    """
    For controlling VideoCapture_Buffers for multiple cameras simultaneously.

    About threading:
    http://pymotw.com/2/threading/index.html#module-threading
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.accumulating = False

    def ListVideoDevices(self, dev_dir='/dev'):
        """
        Looks in dev_dir for video devices (i.e. video*).
        Only for *nix systems.
        """

        tmplist = glob('%s/video*' % (dev_dir.rstrip('/')))
        self.dev_list = []

        if len(tmplist):
            if self.verbose:
                wsp = ' '*9
                print('%s\n\tVideo device listing' % ('-' * 38))
                print('openCV device name | Linux device name\n%s' %
                      ('-' * 38))
            for idev, dev in enumerate(tmplist):
                self.dev_list.append({'VideoCapture_device': idev,
                                      'Linux_device': dev})
                if self.verbose:
                    print('%s%d%s|\t%s' % (wsp, idev, wsp, dev))
        else:
            if self.verbose:
                print('No video devices found in "%s"' % (dev_dir))

    def InitVCBuffers(self,
                      preonset_nf=25,
                      fps=25,
                      VCDevice_list=['/dev/video0'],
                      render_timestamp=True):
        """
        """
        self.preonset_nf = preonset_nf
        self.render_timestamp = render_timestamp
        self.VCBufs = []
        for vc_dev in VCDevice_list:
            self.VCBufs.append(
                VideoCapture_Buffer(preonset_nf=self.preonset_nf,
                                    dev=vc_dev,
                                    fps=fps,
                                    frame_size=(640, 480),
                                    render_timestamp=self.render_timestamp))

    def RunBuffers(self):
        """
        """
        
        self.event = Event()
        self.VCBuf_Threads = []
        self.buffers_running = [False]*len(self.VCBufs)
        self.frame_sizes = []
        
        for trd_ix, vcbuf in enumerate(self.VCBufs):
            self.VCBuf_Threads.append( Thread( target=vcbuf._run,
                                               args=(self.event,) ) )
            self.VCBuf_Threads[-1].start()
            self.buffers_running[trd_ix] = self.VCBuf_Threads[-1].isAlive()
            self.frame_sizes.append({'width':vcbuf.frame_width,
                                     'height':vcbuf.frame_height})

                    
    def StartAccumulation( self ):
        """
        """
        
        self.event.set()
        self.accumulating = True
        
        
    def GetAccumulated( self ):
            """
            """
        
            if not self.accumulating:
                if self.verbose:
                    print("Start accumulating buffer " 
                        "[VideoCapture_BufferControl().StartAccumulation()]\n"
                        "before calling GetAccumulated().")
            else:
            
                self.event.clear()
                self.accumulating = False
                
                
    def StopBuffers( self ):

        for vcbuf in self.VCBufs:
            vcbuf.stop_buffer()


class VideoCapture_Buffer:
    """
    For event triggered video capture. If multiple over-lapping triggers occur
    (i.e. calls to VideoCapture_Buffer().start_accumulation() within
    postoffset_dur), then those will be captured in parallel. I.e. one buffer
    does not need to finish before another buffer can start.

    Uses threading to not block while capturing frames.

    vcb = VCB()

    Methods
    -------
    vcb.run_buffer          -- Starts a continously running buffer of frame
                                   and timestamps.
    vcb.start_accumulation  -- Starts accumulation of frames and timestamps.
    vcb.get_accumulated     -- Stops accumulation
                               (lets it run for postoffset_nf) and puts the
                               data in vcb.frame_buffer and vcb.ts_buffer.
    vcb.flush_buffer        -- If the accululation was started but it was later
                               decided that the frames should not be used.
    vcb.stop_buffer         -- Stops camera and the running capture.


    About threading:
    http://pymotw.com/2/threading/index.html#module-threading

    About fps (frame rate) and frame_size:
    It seems the transfer rate of the USB 2.0 is limiting. So, higher fps
    requires smaller frame_size.

    To figure out which frame sizes and fps are supported by a camera use the
    command (form linux terminal):
    $ lsusb -v

    In the oputput look for sections describing webcams, e.g.:

      Bus 001 Device 028: ID 0458:7088 KYE Systems Corp. (Mouse Systems)
      Device Descriptor:
        bLength                18
        bDescriptorType         1
        bcdUSB               2.00
        bDeviceClass          239 Miscellaneous Device
        bDeviceSubClass         2 ?
        bDeviceProtocol         1 Interface Association
        bMaxPacketSize0        64
        idVendor           0x0458 KYE Systems Corp. (Mouse Systems)
        idProduct          0x7088
        bcdDevice            7.13
        iManufacturer           1 KYE Systems Corp.
        iProduct                2 WideCam 1050
                ...

    If this is the relevant device, then find the subsection
    "VideoStreaming Interface Descriptor", there are several describing the
    available frame rate and size combinations. E.g.:

      VideoStreaming Interface Descriptor:
        bLength                            30
        bDescriptorType                    36
        bDescriptorSubtype                  5 (FRAME_UNCOMPRESSED)
        bFrameIndex                         1
        bmCapabilities                   0x00
          Still image unsupported
        wWidth                            640
        wHeight                           480
        dwMinBitRate                147456000
        dwMaxBitRate                147456000
        dwMaxVideoFrameBufferSize      614400
        dwDefaultFrameInterval         333333
        bFrameIntervalType                  1
        dwFrameInterval( 0)            333333

    Here we can see that the frame_size is wHeight=480 and wWidth=640.
    The frame rate is given by "dwFrameInterval", i.e. the inverse of fps.
    It seems that the FrameInterval is 100ns units
    (https://searchcode.com/codesearch/view/17498588/). At lest it fits.
    I.e. 1/(dwFrameInterval*100*10**-9) will give fps.
    E.g. 1/(333333*100*10**-9) = 30 fps.

    Conclusion.
    It appears as with a USB camera the highest fps for frame_size = (640, 480)
    is 30, and for frame_size = (1280, 720) it is 15 or mabey only 9 fps.

    By Hjalmar K. Turesson, 2014-10-23
    """

    def __init__(self, preonset_dur=1.0, postoffset_dur=0.0,
                 dev='/dev/video0', frame_size=(640, 480), fps=30,
                 render_timestamp=True, verbose=True, save_frames=False):

        #frame_size=(1280, 720)
        pygame.init()
        pygame.camera.init()

        self.frame_size = frame_size
        self.frame_width = self.frame_size[0]
        self.frame_height = self.frame_size[1]
        self.fps = fps
        self.render_timestamp = render_timestamp
        self.preonset_dur = preonset_dur
        self.preonset_nf = int(np.ceil(self.fps * preonset_dur))
        self.postoffset_dur = postoffset_dur
        self.postoffset_nf = int(np.ceil(self.fps * postoffset_dur))
        self.verbose = verbose
        self.accumulating = False
        self._ts_fmt = "Date: %Y-%m-%d, Time: %H:%M:%S.%f"
        self.timestamp = datetime.now()
        if self.render_timestamp:
            self._prepare_render_timestamp()
        self.save_frames = save_frames
        if self.save_frames:
            self.out_fn = 'tmp.npz'

        self.dev = dev
        self.dev_list = pygame.camera.list_cameras()
        if len(glob(dev)):  # Check that selected device exists.
            self.cam = pygame.camera.Camera(self.dev, self.frame_size, 'RGB')
        elif len(self.dev_list) > 0:
            print("Camera device %s doesn't exist.\nTry" % (dev), end=' ')
            if len(self.dev_list) > 1:
                print("one of %s." % (", ".join(self.dev_list)))
            else:
                print(self.dev_list[0])
        else:
            print("No video devices were found on the system.")
            return 0

    def _run(self):
        """
        """
        self.frame_buffer = []
        self.ts_buffer = []
        preonset_frame_buffer = []
        preonset_ts_buffer = []
        local_frame_buffers = []
        local_ts_buffers = []
        nf_more = []
        self.accumulating = False
        self.flush = False
        self.camera_running = True

        while self.camera_running:

            # This is a blocking call that returns
            # when the camera has an image ready.
            self.image = self.cam.get_image()
            self.timestamp = datetime.now()

            self.frame = pygame.surfarray.array3d(self.image).\
                transpose((1, 0, 2))
            if self.render_timestamp:  # Draw the timestamp
                self._render_timestamp()

            # Append the frame and timestamp to local_frame_buffers
            preonset_frame_buffer.append(self.frame)
            preonset_ts_buffer.append(self.timestamp.timestamp())
            for buf_ix in range(len(local_frame_buffers)):
                local_frame_buffers[buf_ix].append(self.frame)
                local_ts_buffers[buf_ix].append(self.timestamp.timestamp())
                if buf_ix < len(nf_more):
                    nf_more[buf_ix] -= 1

            # The oldest buffer is the 1st (local_frame_buffers[0]),
            # and should thus be done 1st
            # When done: pop() and save (if save)
            if len(nf_more) and nf_more[0] < 1:
                self.frame_buffer = local_frame_buffers.pop(0)
                self.ts_buffer = local_ts_buffers.pop(0)
                nf_more.pop(0)
                if self.save_frames:  # Save the video.
                    if self.verbose:
                        print('Saving frames to %s' % self.out_fn)
                    multiproc_save_frames(self.out_fn,
                                          self.frame_buffer,
                                          self.ts_buffer)

            #------------------------------------------------------------------
            #  Once the buffer(s) is running (run_buffer()) there are 3 options
            # (apart from stopping the buffer (stop_buffer())):
            #
            # 1) Start accumulating frames (start_accumulation())
            # 2) Stop accumulating frames (get_accumulated).
            #    From this point, postoffset_nf more frames will be added.
            #    Once the last frame is added, they will be copied over to
            #    self.frame_buffer and, possibly, saved.
            # 3) Flush the buffer (flush_buffer()).
            #    If a buffer was started to accumulate, but this is regretted,
            #    then pop/get rid of it.
            #------------------------------------------------------------------

            # self.start_accumulation()
            # If start accumulation, just keep appending to local_buffers
            if self._event.isSet() and (not self.accumulating):
                self.accumulating = True
                local_frame_buffers.append(preonset_frame_buffer.copy())
                local_ts_buffers.append(preonset_ts_buffer.copy())

            # self.getBuffer()
            elif (not self._event.isSet()) and self.accumulating:
                self.accumulating = False
                nf_more.append(self.postoffset_nf)

            # self.flush_buffer()
            # Ditch/pop the latest added buffer
            elif self.flush and self.accumulating:
                self.accumulating = False
                self.flush = False
                self._event.clear()
                local_frame_buffers.pop(-1)
                local_ts_buffers.pop(-1)

            # Maintain constant length (i.e. len() = preonset_nf)
            preonset_frame_buffer = preonset_frame_buffer[-self.preonset_nf:]
            preonset_ts_buffer = preonset_ts_buffer[-self.preonset_nf:]

    def run_buffer(self):
        """
        """
        self.cam.start()
        self._event = Event()
        self._thread = Thread(target=self._run)
        self._thread.start()
        self.accumulating = False
        self.flush = False

    def start_accumulation(self):
        """
        """
        if self.accumulating:
            if self.verbose:
                print("Already accumulating.")
        else:
            self._event.set()

    def get_accumulated(self):
        """
        """
        if not self.accumulating:
            if self.verbose:
                print("Start accumulating buffer "
                      "[VideoCapture_Buffer().start_accumulation()]\n"
                      "before calling get_accumulated().")
        else:
            self._event.clear()

    def flush_buffer(self):
        """
        If accumulating but we do not want to get the accumulated frames.
        """
        if not self.accumulating:
            if self.verbose:
                print("No buffer flushed since no buffer was accumulating.")
        else:
            self.flush = True

    def stop_buffer(self):
        """
        """
        # NOTE make a nice stopping routine
        self.cam.stop()
        self.camera_running = False
        self.accumulating = False
        if self.verbose:
            print('Stopped camera')

    def _prepare_render_timestamp(self):
        """
        Prepare position, format, size, color and stuff of the timestamp that
        will be written on each frame.
        """

        ts_str = self.timestamp.strftime(self._ts_fmt)[:-3]
        if pygame.font:
            self._tsSpecs = {'text_rect': 0, 'bkg_rect': 0, 'font': 0}
            self._tsSpecs['font'] = pygame.font.Font(None, 23)
            text = self._tsSpecs['font'].render(ts_str, 1, (255, 255, 255))
            txt_width = text.get_size()[0]
            txt_height = text.get_size()[1]
            bkg_width = txt_width * 1.05
            bkg_height = txt_height * 1.2
            bkg_left = 0
            txt_left = bkg_left + (bkg_width - txt_width)/2.0
            bkg_top = 1 + self.frame_height - bkg_height
            txt_top = bkg_top + (bkg_height - txt_height)/2.0

            self._tsSpecs['text_rect'] = Rect(txt_left,
                                              txt_top,
                                              txt_width,
                                              txt_height)
            self._tsSpecs['bkg_rect'] = Rect(bkg_left,
                                             bkg_top,
                                             bkg_width,
                                             bkg_height)

        else:
            print('Timestamp cannot be rendered. Pygame.font is missing.')
            self.render_timestamp = False
            return 0

    def _render_timestamp(self):
        """
        Renders/draws the timestamp on the frame.

        """

        ts_str = self.timestamp.strftime(self._ts_fmt)[:-3]
        text = self._tsSpecs['font'].render(ts_str, 1, (255, 255, 255))
        self.image.fill((0, 0, 0), self._tsSpecs['bkg_rect'])
        self.image.blit(text, self._tsSpecs['text_rect'])


class VideoCapture_Buffer_fc2:
    """
    For event triggered video capture. If multiple over-lapping triggers occur
    (i.e. calls to VideoCapture_Buffer().start_accumulation() within
    postoffset_dur), then those will be captured in parallel. I.e. one buffer
    does not need to finish before another buffer can start.

    Uses threading to not block while capturing frames.

    vcb = VCB()

    Methods
    -------
    vcb.run_buffer          -- Starts a continously running buffer of frame
                                   and timestamps.
    vcb.start_accumulation  -- Starts accumulation of frames and timestamps.
    vcb.get_accumulated     -- Stops accumulation
                               (lets it run for postoffset_nf) and puts the
                               data in vcb.frame_buffer and vcb.ts_buffer.
    vcb.flush_buffer        -- If the accululation was started but it was later
                               decided that the frames should not be used.
    vcb.stop_buffer         -- Stops camera and the running capture.


    Multiple buffers.
    Only the last added buffer (buffers[-1]) can be accumulating,
    all other buffers will be in the postoffset state
    (i.e. accumulating postoffset_nf).
    Only an accumulating (last added buffer) can be flushed.
    Only the first added buffer can be copied to frame_buffer (and saved).


    fps -- frame rate will be 15 frames/second. This is the limit for the fc2
           camera at 1280 x 960 resolution. (sleep(dt) is unnecessary since the
           camera only seems to return images at fps, i.e. c.retrieve_buffer
           (image) will wait/sleep until a new image is ready and not return
           the same image twice.)

    About threading:
    http://pymotw.com/2/threading/index.html#module-threading
    """
    # TODO, NOTE
    # FIX somekind of approx memory check
    # Proper help text
    # check all details

    def __init__(self, preonset_dur=1.0, postoffset_dur=0.0,
                 frame_size=(640, 480), render_timestamp=True,
                 verbose=True, demosaic=True, save_frames=False):

        if 480 in frame_size and 640 in frame_size:
            fc2_frame_size = fc2.VIDEOMODE_640x480Y8
        elif 1280 in frame_size and 960 in frame_size:
            fc2_frame_size = VIDEOMODE_1280x960Y8
        else:
            print('Frame size for flycapture2 camera is not supported.')
            return 0
            
        pygame.init()

        self.c = fc2.Context()
        self.c.connect(*self.c.get_camera_from_index(0))
        self.c.set_video_mode_and_frame_rate(fc2_frame_size,
                                             fc2.FRAMERATE_15)

        p = self.c.get_property(fc2.FRAME_RATE)
        self.c.set_property(**p)
        self.image = fc2.Image()

        self.frame_size = frame_size
        self.frame_width = self.frame_size[0]
        self.frame_height = self.frame_size[1]
        self.fps = p['abs_value']
        self.render_timestamp = render_timestamp
        self.preonset_dur = preonset_dur
        self.preonset_nf = int(np.ceil(self.fps * preonset_dur))
        self.postoffset_dur = postoffset_dur
        self.postoffset_nf = int(np.ceil(self.fps * postoffset_dur))
        self.verbose = verbose
        self.demosaic = demosaic
        if render_timestamp and not self.demosaic:
            print('Cannot render timestamp on a not demosaiced image.')
            print('render_timestamp will be set to False')
            self.render_timestamp = False
        self.accumulating = False
        self._ts_fmt = "Date: %Y-%m-%d, Time: %H:%M:%S.%f"
        self.timestamp = datetime.now()
        if self.render_timestamp:
            self._prepare_render_timestamp()
        self.save_frames = save_frames
        if self.save_frames:
            self.out_fn = 'tmp.npz'

    def _run(self):
        self.frame_buffer = []
        self.ts_buffer = []
        preonset_frame_buffer = []
        preonset_ts_buffer = []
        local_frame_buffers = []
        local_ts_buffers = []
        nf_more = []
        self.accumulating = False
        self.flush = False
        self.camera_running = True

        while self.camera_running:
            self.c.retrieve_buffer(self.image)
            self.timestamp = datetime.now()

            if self.demosaic:  # Demosaic the image
                self.frame = np.fliplr(np.flipud(bayer(np.uint8(
                    np.array(self.image).squeeze()))))
            else:
                self.frame = np.uint8(np.array(self.image).squeeze())
                if self.render_timestamp:  # Draw the timestamp
                    self._render_timestamp()

            # Append the frame and timestamp to local_frame_buffers
            preonset_frame_buffer.append(self.frame)
            preonset_ts_buffer.append(self.timestamp.timestamp())
            for buf_ix in range(len(local_frame_buffers)):
                local_frame_buffers[buf_ix].append(self.frame)
                local_ts_buffers[buf_ix].append(self.timestamp.timestamp())
                if buf_ix < len(nf_more):
                    nf_more[buf_ix] -= 1

            # The oldest buffer is the 1st (local_frame_buffers[0]),
            # and should thus be done 1st
            # When done: pop() and save (if save)
            if len(nf_more) and nf_more[0] < 1:
                self.frame_buffer = local_frame_buffers.pop(0)
                self.ts_buffer = local_ts_buffers.pop(0)
                nf_more.pop(0)
                if self.save_frames:  # Save the video.
                    if self.verbose:
                        print('Saving frames to %s' % self.out_fn)
                    multiproc_save_frames(self.out_fn,
                                          self.frame_buffer,
                                          self.ts_buffer)

            #------------------------------------------------------------------
            #  Once the buffer(s) is running (run_buffer()) there are 3 options
            # (apart from stopping the buffer (stop_buffer())):
            #
            # 1) Start accumulating frames (start_accumulation())
            # 2) Stop accumulating frames (get_accumulated).
            #    From this point, postoffset_nf more frames will be added.
            #    Once the last frame is added, they will be copied over to
            #    self.frame_buffer and, possibly, saved.
            # 3) Flush the buffer (flush_buffer()).
            #    If a buffer was started to accumulate, but this is regretted,
            #    then pop/get rid of it.
            #------------------------------------------------------------------

            # self.start_accumulation()
            # If start accumulation, just keep appending to local_buffers
            if self._event.isSet() and (not self.accumulating):
                self.accumulating = True
                local_frame_buffers.append(preonset_frame_buffer.copy())
                local_ts_buffers.append(preonset_ts_buffer.copy())

            # self.getBuffer()
            elif (not self._event.isSet()) and self.accumulating:
                self.accumulating = False
                nf_more.append(self.postoffset_nf)

            # self.flush_buffer()
            # Ditch/pop the latest added buffer
            elif self.flush and self.accumulating:
                self.accumulating = False
                self.flush = False
                self._event.clear()
                local_frame_buffers.pop(-1)
                local_ts_buffers.pop(-1)

            # Maintain constant length (i.e. len() = preonset_nf)
            preonset_frame_buffer = preonset_frame_buffer[-self.preonset_nf:]
            preonset_ts_buffer = preonset_ts_buffer[-self.preonset_nf:]

    def run_buffer(self):
        """
        """
        self.c.start_capture()
        self._event = Event()
        #self._thread = Thread(target=self._run, args=(self._event,))
        self._thread = Thread(target=self._run)
        self._thread.start()
        self.accumulating = False
        self.flush = False

    def start_accumulation(self):
        """
        """
        if self.accumulating:
            if self.verbose:
                print("Already accumulating.")
        else:
            self._event.set()

    def get_accumulated(self):
        """
        """
        if not self.accumulating:
            if self.verbose:
                print("Start accumulating buffer "
                      "[VideoCapture_Buffer().start_accumulation()]\n"
                      "before calling get_accumulated().")
        else:
            self._event.clear()

    def flush_buffer(self):
        """
        If accumulating but we do not want to get the accumulated frames.
        """
        if not self.accumulating:
            if self.verbose:
                print("No buffer flushed since no buffer was accumulating.")
        else:
            self.flush = True

    def stop_buffer(self):
        """
        """
        # NOTE make a nice stopping routine
        self.c.stop_capture()
        self.camera_running = False
        self.accumulating = False
        if self.verbose:
            print('Stopped camera')

    def disconnect_cam(self):
        if self.camera_running:
            self.stop_buffer()
        self.c.disconnect()
        if self.verbose:
            print('Disconnected camera')

    def _prepare_render_timestamp(self):
        """
        Prepare position, format, size, color and stuff of the timestamp that
        will be written on each frame.
        """

        ts_str = self.timestamp.strftime(self._ts_fmt)[:-3]
        if pygame.font:
            self._tsSpecs = {'text_rect': 0, 'bkg_rect': 0, 'font': 0}
            self._tsSpecs['font'] = pygame.font.Font(None, 23)
            text = self._tsSpecs['font'].render(ts_str, 1, (255, 255, 255))
            txt_width = text.get_size()[0]
            txt_height = text.get_size()[1]
            bkg_width = txt_width * 1.05
            bkg_height = txt_height * 1.2
            bkg_left = self.frame_height - bkg_height
            txt_left = bkg_left + (bkg_height - txt_height)/2.0
            bkg_top = 0
            txt_top = bkg_top + (bkg_width - txt_width)/2.0

            self._tsSpecs['text_rect'] = Rect(txt_top,
                                              txt_left,
                                              txt_width,
                                              txt_height)
            self._tsSpecs['bkg_rect'] = Rect(bkg_top,
                                             bkg_left,
                                             bkg_width,
                                             bkg_height)

        else:
            print('Timestamp cannot be rendered. Pygame.font is missing.')
            self.render_timestamp = False
            return 0

    def _render_timestamp(self):
        """
        Renders/draws the timestamp on the frame.

        """
        ts_str = self.timestamp.strftime(self._ts_fmt)[:-3]
        text = self._tsSpecs['font'].render(ts_str, 1, (255, 255, 255))
        self.pygame_surface = pygame.surfarray. \
            make_surface(self.frame.transpose((1, 0, 2)))
        self.pygame_surface.fill((0, 0, 0), self._tsSpecs['bkg_rect'])
        self.pygame_surface.blit(text, (self._tsSpecs['text_rect'][0],
                                        self._tsSpecs['text_rect'][1]))
        self.frame = pygame.surfarray.array3d(self.pygame_surface). \
            transpose((1, 0, 2))


def send_ttl(durs=100, ittli=0, blocking=False):
    """
    Sends a ttl over the usb port.
    
    NOTE
    TO DO
    
    Parameters
    ----------
    durs     - 
    ittli    -
    blocking -
    
    Returns
    -------
    Nothing
    
    Either sequences separated by ittli ms, or one ttl pulse.
    Sends a sequence of ttls durations over the usb port.
    Pauses ittli (inter-ttl-interval) between the ttls.
    
    When blocking=False:    
    Uses multiprocessing.Process to not block the process/thread from where it
    is called. It will spawn a new process that can run on another cpu.
    
    Reference:
        https://docs.python.org/3.4/library/multiprocessing.html
        http://hackaday.com/2009/09/22/introduction-to-ftdi-bitbang-mode/

    Hjalmar K. Turesson
    2014-10-13
    modified to handle sequences of ttls on 2015-01-27
    """
    # NOTE TODO:
    # write proper help & check state of bitbang before writing to it.
        
    try:
        ittli = float(ittli)
    except TypeError:
        print('TypeError: ittli needs to be a number.')
        return
    
    def ttl_out(durs, ittli):
        """
        """
        with BitBangDevice() as bb:
            for dur in durs:        
                bb.write('0x02')
                sleep(dur/1000.0)
                bb.write('0x08')
                sleep(ittli/1000.0)
                
    if not hasattr(durs, '__iter__'):
        durs = [durs]
    
    if blocking:
        ttl_out(durs, ittli)
    elif not blocking:
        p = Process(target=ttl_out, args=(durs,ittli))
        p.start()


def read_Monitor_log(log_fn):
    """
    Reads log files from at_monitor.

    E.g.:
    log_fn = '~/data/autotrain/monitor/log/call_monitor_log_20140821_14:48:56.csv


    """

    dtypes = [('n', int),
              ('sound_start', 'float'),
              ('sound_end', 'float'),
              ('sound_dur', 'float'),
              ('rec_start', 'float'),
              ('rec_end', 'float'),
              ('rec_dur', 'float'),
              ('type', 'U5'),
              ('sig_fname', 'U60'),
              ('vid_fname', 'U50')]

    data = np.genfromtxt(log_fn, dtype=dtypes,
                         skip_header=3,
                         delimiter=',')

    f = open(log_fn)
    field_names = f.readline().strip().split(',')
    for ix in range(field_names.count(',')):
        field_names.remove(',')
    field_data = f.readline().strip().split(',')
    for ix in range(field_data.count(',')):
        field_data.remove(',')

    f.close()
    time_funk = lambda t: datetime.strptime(t, '%H:%M:%S')
    date_funk = lambda d: datetime.strptime(d, '%Y-%m-%d')

    dtypes = [date_funk, time_funk, np.int, str, np.int,
              np.float, np.float, np.float, str, str, str, str, str]
    exp_params = {}
    for ix, fdata in enumerate(field_data):
        try:
            exp_params[field_names[ix]] = dtypes[ix](fdata)
        except:
            import pdb
            pdb.set_trace()

    return data, exp_params


def read_AT_log(log_fn):
    """
    Reads log files from autotrain.

    E.g.:
    log_fn = '~/data/autotrain/train/log/call_monitor_log_20140821_14:48:56.csv


    """

    f = open(log_fn)
    l0 = f.readline()
    field_names = l0.strip().split(',')
    for ix in range(field_names.count(',')):
        field_names.remove(',')
    field_data = f.readline().strip().split(',')
    for ix in range(field_data.count(',')):
        field_data.remove(',')

    f.close()
    time_funk = lambda t: datetime.strptime(t, '%H:%M:%S')
    date_funk = lambda d: datetime.strptime(d, '%Y-%m-%d')

    if 'reward_delay' in l0 and 'continous_vid_fn' in l0:
        exp_params_dtypes = [date_funk, time_funk, str, np.int, np.float, np.float,
                             np.float, np.float, np.bool, str, str, str, str, str,
                             str]
        data_dtypes = [('n', 'int'),
                       ('call_start', 'float'),
                       ('call_end', 'float'),
                       ('call_dur', 'float'),
                       ('rec_start', 'float'),
                       ('rec_end', 'float'),
                       ('rec_dur', 'float'),
                       ('reward_t', 'float'),
                       ('type', 'U5'),
                       ('reward_dur', 'int'),
                       ('n_pulses', 'int'),
                       ('sig_fname', 'U60'),
                       ('vid_fname', 'U50')]                             
    else:
        exp_params_dtypes = [date_funk, time_funk, str, np.int, np.int, np.float,
                             np.float, np.float, np.bool, str, str, str, str, str]
        data_dtypes = [('n', int),
                       ('call_start', 'float'),
                       ('call_end', 'float'),
                       ('call_dur', 'float'),
                       ('rec_start', 'float'),
                       ('rec_end', 'float'),
                       ('rec_dur', 'float'),
                       ('reward_t', 'float'),
                       ('type', 'U5'),
                       ('n_pulses', 'int'),
                       ('sig_fname', 'U60'),
                       ('vid_fname', 'U50')]          
    exp_params = {}
    for ix, fdata in enumerate(field_data):
        try:
            exp_params[field_names[ix]] = exp_params_dtypes[ix](fdata)
        except:
            import pdb
            pdb.set_trace()

    data = np.genfromtxt(log_fn, dtype=data_dtypes,
                         skip_header=3,
                         delimiter=',')            

    return data, exp_params


def read_AT_experiment_parameters(log_fn):
    """
    """

    f = open(log_fn)
    l0 = f.readline()
    field_names = l0.strip().split(',')
    for ix in range(field_names.count(',')):
        field_names.remove(',')
    field_data = f.readline().strip().split(',')
    for ix in range(field_data.count(',')):
        field_data.remove(',')

    f.close()
    time_funk = lambda t: datetime.strptime(t, '%H:%M:%S')
    date_funk = lambda d: datetime.strptime(d, '%Y-%m-%d')

    if 'reward_delay' in l0 and 'continous_vid_fn' in l0:
        dtypes = [date_funk, time_funk, str, np.int, np.float, np.float,
                  np.float, np.float, np.bool, str, str, str, str, str, str]
    else:
        dtypes = [date_funk, time_funk, str, np.int, np.int, np.float,
                  np.float, np.float, np.bool, str, str, str, str, str]
    exp_params = {}
    for ix, fdata in enumerate(field_data):
        exp_params[field_names[ix]] = dtypes[ix](fdata)

    return exp_params


def read_AT_log_pulsecounts(log_fn):
    """
    Reads log files from autotrain.

    E.g.:
    log_fn = '~/data/autotrain/train/log/AT_log_20141201_10-33-40.csv


    """

    dtypes = [('n', 'int'),
              ('call_start', 'float'),
              ('call_end', 'float'),
              ('call_dur', 'float'),
              ('rec_start', 'float'),
              ('rec_end', 'float'),
              ('rec_dur', 'float'),
              ('reward_t', 'float'),
              ('type', 'U5'),
              ('reward_dur', 'int'),
              ('n_pulses', 'int'),
              ('sig_fname', 'U60'),
              ('vid_fname', 'U50')]

    data = np.genfromtxt(log_fn, dtype=dtypes,
                         skip_header=3,
                         delimiter=',')

    f = open(log_fn)
    field_names = f.readline().strip().split(',')
    for ix in range(field_names.count(',')):
        field_names.remove(',')
    field_data = f.readline().strip().split(',')
    for ix in range(field_data.count(',')):
        field_data.remove(',')

    f.close()
    time_funk = lambda t: datetime.strptime(t, '%H:%M:%S')
    date_funk = lambda d: datetime.strptime(d, '%Y-%m-%d')

    dtypes = [date_funk, time_funk, str, np.int, np.float, np.float,
              np.float, np.float, np.bool, str, str, str, str, str, str]
    exp_params = {}
    for ix, fdata in enumerate(field_data):
        try:
            exp_params[field_names[ix]] = dtypes[ix](fdata)
        except:
            import pdb
            pdb.set_trace()

    return data, exp_params


def AT_logs_2_dataframe(log_dir=None, log_fnames=None, timestamps=False):
    """
    Parameters
    ----------
    log_dir    --
    log_fnames -- list of log file names, default all logs in log_dir
    timestamps -- Switch between the data (CALL_START, CALL_END, REC_START,
                  REC_END, REWARD_T) in timestamp format or as binned?? xxxx???
    """

    if log_dir is None:
        log_dir = '/home/hjalmar/data/autotrain/train/log/'
    if log_fnames is None:
        log_fnames = glob(log_dir+'AT_log_*_*.csv')

    log_fnames.sort()
    n_logs = len(log_fnames)

    call_start = pd.DataFrame()
    call_end = pd.DataFrame()
    reward_t = pd.DataFrame()
    rec_start = pd.DataFrame()
    rec_end = pd.DataFrame()
    call_dur = pd.DataFrame()
    rec_dur = pd.DataFrame()
    n_pulses = pd.DataFrame()

    dates = []
    times = []
    subject_names = []
    subject_numbers = []
    #reward_durs = []

    for fn in log_fnames:
        exp_params = read_AT_experiment_parameters(fn)
        dates.append(exp_params['date'])
        times.append(exp_params['start_time'].time())
        subject_names.append(exp_params['subject_name'])
        subject_numbers.append(exp_params['subject_number'])
        #reward_durs.append(exp_params['reward_dur'])
        data = pd.read_csv(fn, skiprows=2)

        call_dur = call_dur.append(data.call_dur, ignore_index=True)
        rec_dur = rec_dur.append(data.rec_dur, ignore_index=True)
        n_pulses = n_pulses.append(data.n_pulses, ignore_index=True)

        call_start = call_start.append(data.call_start, ignore_index=True)
        call_end = call_end.append(data.call_end, ignore_index=True)
        rec_start = rec_start.append(data.rec_start, ignore_index=True)
        rec_end = rec_end.append(data.rec_end, ignore_index=True)
        reward_t = reward_t.append(data.reward_t, ignore_index=True)

    if timestamps:
            CALL_START = pd.DataFrame(call_start.values, index=dates)
            CALL_END = pd.DataFrame(call_end.values, index=dates)
            REC_START = pd.DataFrame(rec_start.values, index=dates)
            REC_END = pd.DataFrame(rec_end.values, index=dates)
            REWARD_T = pd.DataFrame(reward_t.values, index=dates)
    else:
        max_time = np.int(round(reward_t.max().max()))+1

        ts = call_start.values
        call_start = pd.DataFrame(np.zeros((n_logs, max_time)),
                                  columns=np.arange(max_time)/60,
                                  dtype=np.bool)
        for ix, row in enumerate(ts):
            ts2binned(call_start.values[ix], row)

        ts = call_end.values
        call_end = pd.DataFrame(np.zeros((n_logs, max_time)),
                                columns=np.arange(max_time)/60,
                                dtype=np.bool)
        for ix, row in enumerate(ts):
            ts2binned(call_end.values[ix], row)

        ts = rec_start.values
        rec_start = pd.DataFrame(np.zeros((n_logs, max_time)),
                                 columns=np.arange(max_time)/60,
                                 dtype=np.bool)
        for ix, row in enumerate(ts):
            ts2binned(rec_start.values[ix], row)

        ts = rec_end.values
        rec_end = pd.DataFrame(np.zeros((n_logs, max_time)),
                               columns=np.arange(max_time)/60,
                               dtype=np.bool)
        for ix, row in enumerate(ts):
            ts2binned(rec_end.values[ix], row)

        ts = reward_t.values
        reward_t = pd.DataFrame(np.zeros((n_logs, max_time)),
                                columns=np.arange(max_time)/60,
                                dtype=np.bool)
        for ix, row in enumerate(ts):
            ts2binned(reward_t.values[ix], row)

        index = [['ts']*len(call_start.columns), call_start.columns]
        CALL_START = pd.DataFrame(call_start.values, index=dates,
                                  columns=index)
        CALL_END = pd.DataFrame(call_end.values, index=dates,
                                columns=index)
        REC_START = pd.DataFrame(rec_start.values, index=dates,
                                 columns=index)
        REC_END = pd.DataFrame(rec_end.values, index=dates,
                               columns=index)
        REWARD_T = pd.DataFrame(reward_t.values, index=dates,
                                columns=index)

    REC_DUR = pd.DataFrame(rec_dur.values, index=dates)
    N_PULSES = pd.DataFrame(n_pulses.values, index=dates)
    CALL_DUR = pd.DataFrame(call_dur.values, index=dates)

    CALL_START.insert(loc=0, column='time', value=times)
    CALL_START.insert(loc=1, column='subject_name', value=subject_names)
    CALL_START.insert(loc=2, column='subject_number', value=subject_numbers)
    #CALL_START.insert(loc=3, column='reward_dur', value=reward_durs)

    CALL_END.insert(loc=0, column='time', value=times)
    CALL_END.insert(loc=1, column='subject_name', value=subject_names)
    CALL_END.insert(loc=2, column='subject_number', value=subject_numbers)
    #CALL_END.insert(loc=3, column='reward_dur', value=reward_durs)

    CALL_DUR.insert(loc=0, column='time', value=times)
    CALL_DUR.insert(loc=1, column='subject_name', value=subject_names)
    CALL_DUR.insert(loc=2, column='subject_number', value=subject_numbers)
    #CALL_DUR.insert(loc=3, column='reward_dur', value=reward_durs)

    REC_START.insert(loc=0, column='time', value=times)
    REC_START.insert(loc=1, column='subject_name', value=subject_names)
    REC_START.insert(loc=2, column='subject_number', value=subject_numbers)
    #REC_START.insert(loc=3, column='reward_dur', value=reward_durs)

    REC_END.insert(loc=0, column='time', value=times)
    REC_END.insert(loc=1, column='subject_name', value=subject_names)
    REC_END.insert(loc=2, column='subject_number', value=subject_numbers)
    #REC_END.insert(loc=3, column='reward_dur', value=reward_durs)

    REC_DUR.insert(loc=0, column='time', value=times)
    REC_DUR.insert(loc=1, column='subject_name', value=subject_names)
    REC_DUR.insert(loc=2, column='subject_number', value=subject_numbers)
    #REC_DUR.insert(loc=3, column='reward_dur', value=reward_durs)

    REWARD_T.insert(loc=0, column='time', value=times)
    REWARD_T.insert(loc=1, column='subject_name', value=subject_names)
    REWARD_T.insert(loc=2, column='subject_number', value=subject_numbers)
    #REWARD_T.insert(loc=3, column='reward_dur', value=reward_durs)

    N_PULSES.insert(loc=0, column='time', value=times)
    N_PULSES.insert(loc=1, column='subject_name', value=subject_names)
    N_PULSES.insert(loc=2, column='subject_number', value=subject_numbers)
    #N_PULSES.insert(loc=3, column='reward_dur', value=reward_durs)

    return CALL_START, CALL_END, CALL_DUR, N_PULSES, \
        REC_START, REC_END, REC_DUR, REWARD_T


def ts_2_binned(ts):
    """
    """

    if ts.ndim == 1:
        ix = np.int32(ts.round())
        ix[ix < 0] = 0
        binned = np.zeros(ix[-1]+1, dtype=np.bool)
        binned[ix] = True
    elif ts.ndim == 2:
        binned = np.zeros((ts.shape[0], int(round(ts.max()))+1),
                          dtype=np.bool)
        for row_num, row in enumerate(ts):
            ts2binned(binned[row_num], row)
    else:
        print('Number of dims in "ts" is not supported.')
        return 0

    return binned


def ts2binned(binned, ts):
    ix = np.int32(ts.round())
    ix[ix < 0] = 0
    binned[ix] = True


def dual_video(dur,
               fname,
               fps=15,
               dev1='/dev/video0',
               verbose=True,
               t_start=None):
    """
    """
    if fname[0] == '~':
        fname = '%s%s' % (os.path.expanduser('~'), fname[1:])
    if '.mp4' in fname:
        fname0 = fname[:-4] + '_0' + fname[-4:]
        fname1 = fname[:-4] + '_1' + fname[-4:]
        log_fname = fname[:-4] + '_timestamps.csv'
    else:
        fname0 = fname + '_0.mp4'
        fname1 = fname + '_1.mp4'
        log_fname = fname + '_timestamps.csv'
        
    if fps == 15:
        fc2_fps = fc2.FRAMERATE_15
    elif fps == 30:
        fc2_fps = fc2.FRAMERATE_30
        print('For now the framerate has to be 15 fps. FIX THIS')
        return 0

    ts_log = open(log_fname, mode='w')
    ts_log.write('%s,%s\n' % (fname0.split('/')[-1], fname1.split('/')[-1]))
    cam0 = fc2.Context()
    cam0.connect(*cam0.get_camera_from_index(0))
    cam0.set_video_mode_and_frame_rate(fc2.VIDEOMODE_640x480Y8, fc2_fps)
    p = cam0.get_property(fc2.FRAME_RATE)
    cam0.set_property(**p)
    image0 = fc2.Image()
    
    pygame.init()
    pygame.camera.init()    
    cam1 = pygame.camera.Camera(dev1, (640, 480), 'RGB')
    image1 = pygame.surface.Surface((640, 480), 0)

    if verbose:
        cmd0 = ['avconv',
               '-y', '-s',
               '%dx%d' % (640, 480),
               '-f', 'rawvideo',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-vcodec', 'mpeg4',
               '-b', '3000k',
               ] + [fname0]
    
        cmd1 = ['avconv',
               '-y', '-s',
               '%dx%d' % (640, 480),
               '-f', 'rawvideo',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-vcodec', 'mpeg4',
               '-b', '3000k',
               ] + [fname1]
    else:
        cmd0 = ['avconv',
                '-loglevel',
                'quiet',
               '-y', '-s',
               '%dx%d' % (640, 480),
               '-f', 'rawvideo',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-an',               
               '-c:v', 'libx264',
               '-crf', '18'
               ] + [fname0]
    
        cmd1 = ['avconv',
                '-loglevel',
                'quiet',                
               '-y', '-s',
               '%dx%d' % (640, 480),
               '-f', 'rawvideo',
               '-pix_fmt', 'rgb24',
               '-r', str(fps),
               '-i', '-',
               '-an',
               '-c:v', 'libx264',
               '-crf', '18'
               ] + [fname1]
                   
    p0 = subprocess.Popen(cmd0, stdin=subprocess.PIPE)
    p1 = subprocess.Popen(cmd1, stdin=subprocess.PIPE)
       
    cam1.start()
    cam0.start_capture()

    t0 = datetime.now().timestamp()
    if not t_start:
        t_start = t0
        
    t = datetime.now().timestamp() - t0
    while t <= dur:
                
        t = datetime.now().timestamp() - t0
        frame0 = np.array(cam0.retrieve_buffer(image0)).squeeze()
        timestamp0 = datetime.now().timestamp() - t_start
        frame0 = np.fliplr(np.flipud(bayer(frame0)))
        
        frame1 = pygame.surfarray.array3d(cam1.get_image(image1)).transpose((1, 0, 2))
        timestamp1 = datetime.now().timestamp() - t_start

        # save
        p0.stdin.write(frame0.tostring())
        p1.stdin.write(frame1.tostring())

        ts_log.write('%s,%s\n' % (timestamp0, timestamp1))
                        
        if verbose:
            t_left = dur - t
            min_str = '{:>3}'.format(str(int(t_left // 60)))
            s_str = '{:>2}'.format(str(int(t_left % 60)))
            t_str = '\033[95m%s min %s s remaining\033[0m' % (min_str, s_str)
            sys.stdout.write(t_str)
            sys.stdout.flush()
            sys.stdout.write('\b'*len(t_str))
            
    ts_log.close()

    p0.stdin.close()
    p0.wait()
    p1.stdin.close()
    p1.wait()
    
    cam1.stop()
    pygame.quit() 
    
    cam0.stop_capture()
    cam0.disconnect()

    
def cont_vid(stop_datetime, start_datetime = None, pause_time=[18, 4], dur=300):
    """
    start_datetime  : format "%Y%m%d_%H:%M:%S", e.g. '20150224_14-03-22'
    stop_datetime   : format "%Y%m%d_%H:%M:%S"
    pause_time      : [hour_off, hour_on], 0-24 h
    """
    if start_datetime is None:
        start_datetime = time.strftime("%Y%m%d_%H:%M:%S")
        
    d = '/home/hjalmar/Dropbox/data/video/cage_monitor/'
    #dur = 60 * 60 # 1 h

    t_start = time.strptime(start_datetime, "%Y%m%d_%H:%M:%S")
    t_stop = time.strptime(stop_datetime, "%Y%m%d_%H:%M:%S")
    t = time.localtime()
    while time.mktime(t) < time.mktime(t_stop):
        if time.mktime(t) > time.mktime(t_start):
            if t.tm_hour < pause_time[0] and t.tm_hour > pause_time[1]:
                fn = "video_cage03_" + time.strftime("%Y%m%d_%H-%M-%S") + '.mp4'
                reader = imageio.get_reader('<video0>')
                writer = imageio.get_writer(d + fn, fps=30)
                
                t0 = perf_counter()
                for im in reader:
                    t_left = dur - (perf_counter() - t0)
                    sleep(1/33)
                    writer.append_data(im)
                    
                    min_str = '{:>3}'.format(str(int(t_left // 60)))
                    s_str = '{:>2}'.format(str(int(t_left % 60)))
                    t_str = '\033[95m%s min %s s remaining\033[0m' % (min_str, s_str)
                    sys.stdout.write(t_str)
                    sys.stdout.flush()
                    sys.stdout.write('\b'*len(t_str))
                    
                    #print('remaining time %d' % (perf_counter() - t0 - dur))
                    if t_left < 0:
                        break    
                writer.close()
                reader.close()
        t = time.localtime()

#
#def gst_test():
#    
#    
#    args = ['gst-launch-1.0']
#    args.extend(("v4l2src device=/dev/video0 ! video/x-h264, width=640, height=480, framerate=20/1 ! filesink location=/dev/stdout").split())
#        
#    cmd = ['avconv',
#           '-y', '-s',
#           '640x480',
#           '-vcodec', 'h264',
#           '-r', str(20),
#           '-i', '-',
#           '-vcodec', 'copy',
#           ] + ['test_h264.mp4']
#           
#    p_in = sp.Popen(args,stdout=sp.PIPE)
#    p_out = sp.Popen(cmd, stdin=sp.PIPE)
#    
#    for i in range(20*60):
#        fr = p_in.stdout.read(614400)
#        p_out.stdin.write(fr)
#    p_in.stdout.close()
#    p_out.stdin.close()
#    p_in.wait()
#    p_out.wait()
        
#==============================================================================
# 
# class Camera:
#     """
#     v4l2-ctl:
#     http://ivtvdriver.org/index.php/V4l2-ctl
#     
#     v4l2-ctl --get-fmt-video
#     
#     gst-launch-1.0 -v -e v4l2src device=/dev/video0 ! queue ! video/x-h264,width=1280,height=720,framerate=30/1 ! \
#       h264parse ! avdec_h264 ! xvimagesink sync=false
#       
#     gst-launch-1.0 -v -e uvch264src device=/dev/video1 name=src auto-start=true \
#         src.vfsrc ! queue ! video/x-raw,format=\(string\)YUY2,width=320,height=240,framerate=10/1 ! xvimagesink sync=false \
#         src.vidsrc ! queue ! video/x-h264,width=1280,height=720,framerate=30/1 ! h264parse ! avdec_h264 ! xvimagesink sync=false      
#         
#      	args = ['gst-launch-1.0']
#        args.extend(("v4l2src device=/dev/video0 ! video/x-h264, width=640, height=480, framerate=20/1 ! filesink location=/dev/stdout").split())
# 	args.extend(("v4l2src device=/dev/video0 ! video/x-h264, width=640, height=480, framerate=20/1 ! h264parse ! avdec_h264 ! filesink location=/dev/stdout").split())
#  
#  
#  	args.extend(("v4l2src device=%s ! video/x-raw-yuv,width=640,height=480 ! ffmpegcolorspace ! theoraenc ! oggmux ! filesink location=%s/camera.ogv").split())
# 	gst_process = subprocess.Popen(args)
#  
#         cmd = ["gst-launch-1.0", "-v", "-e", "v4l2src", "device=/dev/video0", "!", "queue", "!", "video/x-h264", "width=1280", "height=720", "framerate=30/1", "!", "h264parse", "!", "avdec_h264", "!", "filesink location=/dev/stdout" "sync=false"]
#         cmd = ['gst-launch-1.0', '-v', '-e', 'v4l2src', 'device=/dev/video0', '!', 'queue', '!', 'video/x-h264', 'width=1280', 'height=720', 'framerate=30/1', '!', 'h264parse', '!', 'avdec_h264']
#         cmd = ['gst-launch-1.0', '-v', '-e', 'v4l2src', 'device=/dev/video0']
#     """
#     # v4l2-ctl --list-formats
# # v4l2-ctl --list-formats-ext
# # v4l2-ctl --set-fmt-video=width=1280,height=720,pixelformat=1
# # v4l2-ctl --set-parm=30
#     
#     def _initialize():
#         
#     
#                 # Output args, for writing to pipe
#             oargs = ['-f', 'image2pipe',
#                      '-pix_fmt', self._pix_fmt,
#                      '-vcodec', 'rawvideo']
#             oargs.extend(['-s', self._arg_size] if self._arg_size else [])
#             # Create process
#             cmd = ['avconv'] + iargs + ['-i', self._filename] + oargs + ['-']
#             self._proc = sp.Popen(cmd, stdin=sp.PIPE,
#                                   stdout=sp.PIPE, stderr=sp.PIPE)
#==============================================================================

#gst-launch-1.0 -v -e v4l2src device=/dev/video0 ! queue ! video/x-h264,width=1280,height=720,framerate=30/1 ! h264parse ! filesink location='test_h264.mp4'        


class Webcam_h264:
    def __init__(self, video_dev='/dev/video0',
                 audio_dev = None,
                 fps=30,
                 t_start=0.0):
        """
        Understand audio_dev:
            hw:X,Y comes from this mapping of audio hardware 
            X is the card number, while Y is the device number. 
            Use 'arecord -l' to list the available cards and devices.
            http://jan.newmarch.name/LinuxSound/Sampled/Alsa/
        """

        if audio_dev is None:
            audio = False
        elif audio_dev is 'default':
            audio_dev = "hw:2,0"
            audio = True
            
        
        ts_log_fname = 'webcam_h264_timestamps.log'
        vid_fname = 'webcam_h264.mkv'
        self.ts_log = open(ts_log_fname, 'w')
        self.ts_log.write('video filename: %s, t_start: %0.9f'
                          '\nframe_number, offset_ts, pts, cam_running_ts\n' %
                          (vid_fname, t_start))
        if not t_start:
            self.t_start = datetime.now().timestamp()
        else:
            self.t_start = t_start

        self.n_frames = 0

        # Create GStreamer pipline
        self.pipeline = Gst.Pipeline()

        # Create bus to get events from GStreamer pipeline
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message::error', self.on_error)

        # This is needed to make the video output in our DrawingArea:
        self.bus.enable_sync_message_emission()
        self.bus.connect('sync-message::element', self.on_sync_message)

        ###########################
        # Callable function
        ###########################
        def on_new_sample(appsink):
            """
            Function called from the pipeline by appsink.
            Writes the timestampes of frame capture to a log file.
            """
            # Get the buffer
            smp = appsink.emit('pull-sample')
            buf = smp.get_buffer()
            timestamp = np.float64(1e-9) * buf.pts + self.offset_t
            timestamp1 = datetime.now().timestamp() - self.t_start
            self.n_frames += 1
            self.ts_log.write('%d,%0.9f,%0.9f,%0.9f\n' % 
                              (self.n_frames,
                               timestamp,
                               np.float64(1e-9) * buf.pts,
                               timestamp1))
            return False

        ###########################
        # Create GStreamer elements
        ###########################
        # Video source:
        self.v4l2src0 = Gst.ElementFactory.make('v4l2src', None)
        self.v4l2src0.set_property('device', video_dev)
        self.v4l2src0.set_property('do-timestamp', 'true')
        # Video source filters:
        vid0caps = Gst.Caps.from_string('video/x-h264,width=%d,height=%d,'
                                        'framerate=%d/1' % (1280, 720, fps))
        self.vid0filter = Gst.ElementFactory.make('capsfilter', None)
        self.vid0filter.set_property('caps', vid0caps)
        # Parse video:
        self.vid0parse = Gst.ElementFactory.make('h264parse', None)
        # Split:
        self.tee0 = Gst.ElementFactory.make('tee', None)
        self.tee0.set_property('name', 't0')
        ####
        # Display branch
        ####
        # Decode
        self.vid0decode = Gst.ElementFactory.make('avdec_h264', None)
        # Scale to display size:
        self.disp0scale = Gst.ElementFactory.make('videoscale', None)
        # Display filter caps:
        disp0caps = Gst.Caps.from_string('video/x-raw,width=%d,height=%d' %
                                         (800, 600))
        # Sinks:
        self.disp0sink = Gst.ElementFactory.make('autovideosink', None)
        self.disp0sink.set_property('filter-caps', disp0caps)
        ####
        # File branch
        ####
        self.mux = Gst.ElementFactory.make('matroskamux', None)
        self.file0sink = Gst.ElementFactory.make('filesink', None)
        self.file0sink.set_property('location', vid_fname)
        self.file0sink.set_property('sync', False)
        ####
        # Timestamp branch
        ####
        # Create appsink
        self.ts0sink = Gst.ElementFactory.make('appsink', None)
        # Setting properties of appsink
        ts0caps = vid0caps  # use same caps as for camera
        self.ts0sink.set_property('caps', ts0caps)
        self.ts0sink.set_property("max-buffers", 20)  # Limit memory usage
        # Tell sink to emit signals
        self.ts0sink.set_property('emit-signals', True)
        self.ts0sink.set_property('sync', False)  # No sync
        # Connect appsink to my function (writing timestamps)
        self.ts0sink.connect('new-sample', on_new_sample)

        self.queue0 = Gst.ElementFactory.make('queue', None)
        self.queue1 = Gst.ElementFactory.make('queue', None)                
        self.disp_queue = Gst.ElementFactory.make('queue', None)
        self.file_queue = Gst.ElementFactory.make('queue', None)
        self.ts_queue = Gst.ElementFactory.make('queue', None)
        
        #self.queue = Gst.ElementFactory.make('queue', None)
        
        if audio:
            # Audio source:
            self.alsasrc0 = Gst.ElementFactory.make('alsasrc')
            self.alsasrc0.set_property('device', audio_dev)
            # Audio source filters:
            aud0caps = Gst.Caps.from_string('audio/x-raw,'
                                            'format=S16LE'
                                            'rate=44100,'
                                            'channels=1')
            self.aud0filter = Gst.ElementFactory.make('capsfilter', None)
            self.aud0filter.set_property('caps', aud0caps)
            # Encode audio:
            self.audconv = Gst.ElementFactory.make('audioconvert', None)
            self.audenc = Gst.ElementFactory.make('flacenc', None)
            self.aud_queue = Gst.ElementFactory.make('queue', None)
        
        # Add elements to the pipeline
        self.pipeline.add(self.v4l2src0)
        self.pipeline.add(self.vid0filter)
        self.pipeline.add(self.vid0parse)
        self.pipeline.add(self.tee0)
        self.pipeline.add(self.vid0decode)
        self.pipeline.add(self.disp0scale)
        self.pipeline.add(self.disp0sink)
        self.pipeline.add(self.mux)
        self.pipeline.add(self.file0sink)
        self.pipeline.add(self.ts0sink)
        self.pipeline.add(self.queue0)
        self.pipeline.add(self.queue1)
        self.pipeline.add(self.disp_queue)
        self.pipeline.add(self.file_queue)
        self.pipeline.add(self.ts_queue)
        
        if audio:
            self.pipeline.add(self.alsasrc0)            
            self.pipeline.add(self.aud0filter)
            self.pipeline.add(self.audconv)
            self.pipeline.add(self.audenc)            
            self.pipeline.add(self.aud_queue)

        ###############
        # Link elements
        ###############
        # video source
        if not self.v4l2src0.link(self.vid0filter):
            print('video source to video filter link failed')          
        if not self.vid0filter.link(self.vid0parse):
            print('video filter to video parse link failed')
        if not self.vid0parse.link(self.tee0):
            print('video parse to tee link failed')
        if audio:
            # audio source
            if not self.alsasrc0.link(self.aud0filter):
                print('audio source to audio filter link failed')
            if not self.aud0filter.link(self.audconv):
                print('audio filter to audio convert link failed')
            if not self.audconv.link(self.audenc):
                print('audio convert to audio enconder link failed')
            if not self.audenc.link(self.aud_queue):
                print('audio enconder to audio queue link failed')      
        # tee
        if not self.tee0.link(self.disp_queue):
            print('tee to display queue link failed')
        if not self.tee0.link(self.file_queue):
            print('tee to file queue link failed')
        if not self.tee0.link(self.ts_queue):
            print('tee to ts queue link failed')
        # video display sink
        if not self.disp_queue.link(self.vid0decode):
            print('dispaly queue to video decode link failed')
        if not self.vid0decode.link(self.disp0scale):
            print('decode to videoscale link failed')
        if not self.disp0scale.link(self.queue0):
            print('disp0scale to queue0 link failed')            
        if not self.queue0.link_filtered(self.disp0sink, disp0caps):
            print('queue0 to display-sink link failed')
        # file sink
        if not self.file_queue.link(self.mux):
            print('file queue to mux link failed')
        if audio:
            if not self.aud_queue.link(self.mux):
                print('audio queue to mux link failed')            
        if not self.mux.link(self.queue1):
            print('mux to queue1 link failed')            
        if not self.queue1.link(self.file0sink):
            print('queue1 to file-sink link failed')
        # timestamp sink
        if not self.ts_queue.link(self.ts0sink):
            print('ts queue to ts-sink link failed')

    def run(self):
        self.offset_t = datetime.now().timestamp() - self.t_start
        self.pipeline.set_state(Gst.State.PLAYING)

    def quit(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.ts_log.close()

    def on_sync_message(self, bus, msg):
        if msg.get_structure().get_name() == 'prepare-window-handle':
            msg.src.set_property('force-aspect-ratio', True)

    def on_error(self, bus, msg):
        print('on_error():', msg.parse_error())


class Webcam:
    """
    Test commands:
    
    raw, mjpg or h264 from camera
    ----------------
    Single to display:
    gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-h264,width=640,height=480,framerate=15/1 ! h264parse ! avdec_h264 ! xvimagesink sync=false
    Dual to display and file:
    gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-h264,width=640,height=480,framerate=15/1 ! tee name=t t. ! queue ! h264parse ! avdec_h264 ! xvimagesink sync=false t. ! queue ! h264parse ! matroskamux ! filesink location='h264_dual.mkv' sync=false
    
    Raw from camera
    ---------------
    Single to display:
    gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! xvimagesink sync=false
    Single to encoded to file:
    gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! videoconvert ! x264enc ! matroskamux ! filesink location='raw_single.mkv' sync=false
    Dual to display and encoded to file:
    gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! tee name=t t. ! queue ! xvimagesink sync=false t. ! queue ! videoconvert ! x264enc ! h264parse ! matroskamux ! filesink location='raw_dual.mkv' sync=false
    gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! tee name=t t. ! queue ! xvimagesink sync=false t. ! queue ! videoconvert ! x264enc tune=zerolatency ! h264parse ! matroskamux ! filesink location='raw_dual.mkv' sync=false
    
    gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! tee name=t t. ! queue ! xvimagesink sync=false t. ! queue ! videoconvert ! theoraenc ! theoraparse ! matroskamux ! filesink location='raw_dual.mkv' sync=false
    
    gst-launch-1.0 -v v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! tee name=t t. ! queue ! xvimagesink sync=false t. ! queue ! videoconvert ! vp8enc ! matroskamux ! filesink location='raw_dual.mkv' sync=false
    
    Good info on v4l2src and webcam stream:
    http://blog.buberel.org/2008/03/but-wait-theres.html
    
    Textoverlay / draw_timestamp info:
    http://stackoverflow.com/questions/22469913/gstreamer-textoverlay-is-not-dynamically-updated-during-play
    https://blogs.gnome.org/uraeus/2012/11/08/gstreamer-python-and-videomixing/
    
    """
    def __init__(self, video_dev='/dev/video0',
                 audio_dev=None,
                 fps=15,
                 t_start=0.0,
                 write=True,
                 display=True,
                 timestamp=True,
                 srcenc='mjpeg',
                 writeenc='theora',
                 draw_timestamp=True):
        """
        Understand audio_dev:
            hw:X,Y comes from this mapping of audio hardware
            X is the card number, while Y is the device number.
            Use 'arecord -l' to list the available cards and devices.
            
        writeenc  - theora, vp8 or h.264
        srcenc   - raw or 
        """
        ts_log_fname = 'webcam_timestamps.log'
        vid_fname = 'webcam.mkv'
        
        self.timestamp=timestamp
        self.write=write
        self.display=display
        if not self.timestamp:
            draw_timestamp = False
        self.draw_timestamp = draw_timestamp
            
        if audio_dev is None:
            self.audio = False
        elif audio_dev is 'default':
            audio_dev = "hw:2,0"
            self.audio = True

        writeparse = None
        writeenc_prop = None
        if 'theora' in writeenc:
            writeenc = 'theoraenc'
            writeenc_caps = 'video/x-theora'
            writeparse = 'theoraparse'
        elif 'vp8' in writeenc:
            writeenc = 'vp8enc'
            writeenc_caps = 'video/x-vp8'
        elif '264' in writeenc:
            writeenc = 'x264enc'
            writeenc_caps = 'video/x-h264'
            writeparse = 'h264parse'
            writeenc_prop = ('tune', 'zerolatency')

        srcparse = None
        srcdec = None
        if 'raw' in srcenc:
            src_caps = ('video/x-raw,'
                         'format=YUY2,'
                         'width=%d,height=%d,'
                         'framerate=%d/1' % (640, 480, fps))
        elif 'peg' in srcenc:
            src_caps = ('image/jpeg,'
                         'width=%d,height=%d,'
                         'framerate=%d/1' % (640, 480, fps))
            srcdec = 'jpegdec'
        elif '264' in srcenc:
            src_caps = ('video/x-h264,'
                         'width=%d,height=%d,'
                         'framerate=%d/1' % (640, 480, fps))
            srcparse = 'h264parse'
            srcdec = 'avdec_h264'
        
        skip_encdec = False
        if ('264' in srcenc) and (writeenc is 'x264enc'):
            skip_encdec = True
            srcdec = None
        
        if self.timestamp:
            self.ts_log = open(ts_log_fname, 'w')
            self.ts_log.write('video filename: %s, t_start: %0.9f'
                              '\nframe_number, offset_ts, cam_running_ts, python_ts\n' %
                              (vid_fname, t_start))
        if not t_start:
            self.t_start = datetime.now().timestamp()
        else:
            self.t_start = t_start

        self.n_frames = 0

        # Create GStreamer pipline
        self.pipeline = Gst.Pipeline()
        # Create bus to get events from GStreamer pipeline
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message::error', self.on_error)
        # This is needed to make the video output in our DrawingArea:
        self.bus.enable_sync_message_emission()
        self.bus.connect('sync-message::element', self.on_sync_message)

        ###########################
        # Create GStreamer elements
        ###########################
        # Video source:
        self.v4l2src0 = Gst.ElementFactory.make('v4l2src', None)
        self.v4l2src0.set_property('device', video_dev)
        self.v4l2src0.set_property('do-timestamp', 'true')
        # Video source filters
        # Formats available from C920 camera: 
        #   'image/jpeg', 'video/x-h264', 'video/x-raw'
        vid_caps = Gst.Caps.from_string(src_caps)
        self.vid_filter = Gst.ElementFactory.make('capsfilter', None)
        self.vid_filter.set_property('caps', vid_caps)
        if not srcparse is None:
            self.src_parse = Gst.ElementFactory.make(srcparse, None)
        if not srcdec is None:
            self.read_dec = Gst.ElementFactory.make(srcdec, None)
        # Split:
        self.tee0 = Gst.ElementFactory.make('tee', None)
        ####
        # File branch
        ####
        if self.write:
            # Encode
            self.file_queue = Gst.ElementFactory.make('queue', None)
            if not skip_encdec:
                self.vid_conv = Gst.ElementFactory.make('videoconvert')
                self.vid_enc = Gst.ElementFactory.make(writeenc, None)
                if not writeenc_prop is None:
                    self.vid_enc.set_property(writeenc_prop[0], writeenc_prop[1])
                if not writeparse is None:
                    self.vid_parse = Gst.ElementFactory.make(writeparse, None)
            self.mux = Gst.ElementFactory.make('matroskamux', None)
            mux_caps = Gst.Caps.from_string('%s,'
                                            'width=%d,height=%d,'
                                            'framerate=%d/1' % 
                                            (writeenc_caps, 640, 480, fps))
            self.file_sink = Gst.ElementFactory.make('filesink', None)
            self.file_sink.set_property('location', vid_fname)
            self.file_sink.set_property('sync', False)

        ####
        # Timestamp branch
        ####
        if self.timestamp:
            if self.draw_timestamp:
                self.textoverlay = Gst.ElementFactory.make('textoverlay', None)
                #self.textoverlay.set_property("halign", "left")
                #self.textoverlay.set_property("valign", "top")
                #self.textoverlay.set_property("shaded-background", "true")
            self.ts_queue = Gst.ElementFactory.make('queue', None)
            # Create appsink
            self.ts_sink = Gst.ElementFactory.make('appsink', None)
            # Tell sink to emit signals
            self.ts_sink.set_property('emit-signals', True)
            self.ts_sink.set_property('sync', False)  # No sync
            self.ts_sink.set_property('drop', False)
            # Connect appsink to my function (writing timestamps)
            self.ts_sink.connect('new-sample', self.on_new_sample)
                            
        ####
        # Display branch
        ####
        if self.display:
            self.disp_queue = Gst.ElementFactory.make('queue', None)
            if skip_encdec:
                self.disp_dec = Gst.ElementFactory.make('avdec_h264', None)
            # Scale to display size:
            self.disp_scale = Gst.ElementFactory.make('videoscale', None)
            # Display filter caps:
            disp_caps = Gst.Caps.from_string('video/x-raw,'
                                              'width=%d,height=%d' %
                                              (640, 480))
            # display sink:
            self.disp_sink = Gst.ElementFactory.make('xvimagesink', None)
            self.disp_sink.set_property('sync', False)
            #self.disp0sink.set_property('filter-caps', disp0caps)
                    
        if self.audio:
            # Audio source:
            self.alsasrc0 = Gst.ElementFactory.make('alsasrc')
            self.alsasrc0.set_property('device', audio_dev)
            # Audio source filters:
            aud_caps = Gst.Caps.from_string('audio/x-raw,'
                                            'format=S16LE'
                                            'rate=44100,'
                                            'channels=1')
            self.aud_filter = Gst.ElementFactory.make('capsfilter', None)
            self.aud_filter.set_property('caps', aud_caps)
            # Encode audio:
            self.audconv = Gst.ElementFactory.make('audioconvert', None)
            self.audenc = Gst.ElementFactory.make('flacenc', None)
            self.aud_queue = Gst.ElementFactory.make('queue', None)
        
        # Add elements to the pipeline
        self.pipeline.add(self.v4l2src0)
        self.pipeline.add(self.vid_filter)
        if not srcparse is None:
            self.pipeline.add(self.src_parse)
        if not srcdec is None:
            self.pipeline.add(self.read_dec)
        self.pipeline.add(self.tee0)
                
        if self.write:
            self.pipeline.add(self.file_queue) 
            if not skip_encdec:
                self.pipeline.add(self.vid_conv)
                self.pipeline.add(self.vid_enc)
                if not writeparse is None:
                    self.pipeline.add(self.vid_parse)
            self.pipeline.add(self.mux)
            self.pipeline.add(self.file_sink)        
        if self.timestamp:
            if self.draw_timestamp:
                self.pipeline.add(self.textoverlay)                
            self.pipeline.add(self.ts_sink)
            self.pipeline.add(self.ts_queue)
        
        if self.display:
            self.pipeline.add(self.disp_scale)
            if skip_encdec:
                self.pipeline.add(self.disp_dec)
            self.pipeline.add(self.disp_sink)
            self.pipeline.add(self.disp_queue)
        
        if self.audio:
            self.pipeline.add(self.alsasrc0)            
            self.pipeline.add(self.aud_filter)
            self.pipeline.add(self.aud_conv)
            self.pipeline.add(self.aud_enc)            
            self.pipeline.add(self.aud_queue)

        ###############
        # Link elements
        ###############
        # video source
        if not self.v4l2src0.link(self.vid_filter):
            print('video source to video filter link failed')
        if not srcparse is None:
            if not self.vid_filter.link(self.src_parse):
                print('video filter to video parse link failed')
            if not srcdec is None:
                if not self.src_parse.link(self.read_dec):
                    print('video parse to video decode link failed')
                if not self.read_dec.link(self.tee0):
                    print('video decode to tee link failed')
            else:
                if not self.src_parse.link(self.tee0):
                    print('video parse to video decode link failed')
        else:
            if not srcdec is None:
                if not self.vid_filter.link(self.read_dec):
                    print('video filter to video decode link failed')
                if not self.read_dec.link(self.tee0):
                    print('video decode to tee link failed')                        
            else:
                if not self.vid_filter.link(self.tee0):
                    print('video filter to tee link failed')

        if self.write:
            if not self.tee0.link(self.file_queue):
                print('tee to file queue link failed')
            if self.draw_timestamp:
                if not self.file_queue.link(self.textoverlay):
                    print('file queue to textoverlay link failed')
                if not skip_encdec:
                    if not self.textoverlay.link(self.vid_conv):
                        print('textoverlay to video converter link failed')
                    if not self.vid_conv.link(self.vid_enc):
                        print('video converter to video encoder link failed')
                    if not writeparse is None:
                        if not self.vid_enc.link(self.vid_parse):
                            print('video encoder to video parser link failed')                  
                        if not self.vid_parse.link_filtered(self.mux, mux_caps):
                            print('video encoder to filter and mux link failed')
                    else:
                        if not self.vid_enc.link_filtered(self.mux, mux_caps):
                            print('video encoder to filter and mux link failed')
                else:
                    if not self.textoverlay.link_filtered(self.mux, mux_caps):
                        print('textoverlay to filter and mux link failed')
            else:
                if not skip_encdec:
                    if not self.file_queue.link(self.vid_conv):
                        print('file queue to video converter link failed')
                    if not self.vid_conv.link(self.vid_enc):
                        print('video converter to video encoder link failed')
                    if not writeparse is None:
                        if not self.vid_enc.link(self.vid_parse):
                            print('video encoder to video parser link failed')                  
                        if not self.vid_parse.link_filtered(self.mux, mux_caps):
                            print('video encoder to filter and mux link failed')
                    else:
                        if not self.vid_enc.link_filtered(self.mux, mux_caps):
                            print('video encoder to filter and mux link failed')
                else:
                    if not self.file_queue.link_filtered(self.mux, mux_caps):
                        print('file queue to filter and mux link failed')
            if not self.mux.link(self.file_sink):
                print('mux to file-sink link failed')

        if self.timestamp:
            # timestamp sink
            if not self.tee0.link(self.ts_queue):
                print('tee to ts queue link failed')
            if not self.ts_queue.link(self.ts_sink):
                print('ts queue to ts-sink link failed')

        if self.display:
            # video display sink
            if not self.tee0.link(self.disp_queue):
                print('tee to display queue link failed')
            if skip_encdec:
                if not self.disp_queue.link(self.disp_dec):
                    print('display queue to display decode link failed')
                if not self.disp_dec.link(self.disp_scale):
                    print('display decode to display scale link failed')
            else:
                if not self.disp_queue.link(self.disp_scale):
                    print('display queue to display scale link failed')                 
            if not self.disp_scale.link_filtered(self.disp_sink, disp_caps):
                print('display scale to display-sink link failed')
                        
        if self.audio:
            # audio source
            if not self.alsasrc0.link(self.aud_filter):
                print('audio source to audio filter link failed')
            if not self.aud_filter.link(self.aud_conv):
                print('audio filter to audio convert link failed')
            if not self.aud_conv.link(self.aud_enc):
                print('audio convert to audio enconder link failed')
            if not self.aud_enc.link(self.aud_queue):
                print('audio enconder to audio queue link failed')
            if not self.aud_queue.link(self.mux):
                print('audio queue to mux link failed')
                
    ###########################
    # Callable function for writing timestamps
    ###########################
    def on_new_sample(self, appsink):
        """
        Function called from the pipeline by appsink.
        Writes the timestampes of frame capture to a log file.
        """
        # Get the buffer
        smp = appsink.emit('pull-sample')
        buf = smp.get_buffer()
        timestamp = np.float64(1e-9) * buf.pts + self.offset_t
        ts1 = datetime.now().timestamp() - self.t_start
        self.n_frames += 1
        self.ts_log.write('%d,%0.9f,%0.9f,%0.9f\n' % 
                          (self.n_frames,
                          timestamp,
                          np.float64(1e-9) * buf.pts,
                          ts1))
        if self.draw_timestamp:
            parsed_ts = ('frame #: %d, ts: %0.5f, pts: %0.5f, ts1: %0.5f\n' % 
                         (self.n_frames,
                         timestamp,
                         np.float64(1e-9) * buf.pts,
                         ts1))
            self.textoverlay.set_property('text', parsed_ts)
        return Gst.FlowReturn.OK

    ######################################################
    # Generates the dot file, checks that graphviz in installed
    # and generates a png file, which it then displays the pipeline
    ######################################################
    def on_debug_activate(self):
        fn = 'pipeline-debug-graph'
        fn_dot = "%s/%s.dot" % (os.environ.get("GST_DEBUG_DUMP_DOT_DIR"), fn)
        fn_png = "%s/%s.png" % (os.environ.get("GST_DEBUG_DUMP_DOT_DIR"), fn)
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, fn)
        try:
            os.system("dot -Tpng %s > %s" % (fn_dot, fn_png))
            print('Pipline graph written to %s' % fn_png)
            Gtk.show_uri(None, "file://%s" % fn_png, 0)
        except:
            print('Failure!')
            # check if graphviz is installed with a simple test
            if os.system('which dot'):
                print("Graphviz does not seem to be installed.")

    def run(self):
        self.offset_t = datetime.now().timestamp() - self.t_start
        self.pipeline.set_state(Gst.State.PLAYING)

    def quit(self):
        self.pipeline.set_state(Gst.State.NULL)
        if self.timestamp:
            self.ts_log.close()

    def on_sync_message(self, bus, msg):
        if msg.get_structure().get_name() == 'prepare-window-handle':
            msg.src.set_property('force-aspect-ratio', True)

    def on_error(self, bus, msg):
        print('on_error():', msg.parse_error())


class Webcam_ts_minimal_h264:
    def __init__(self, video_dev='/dev/video0', fps=15):
        """
        """
        ts_log_fname = 'ts_test.log'
                 
        self.ts_log = open(ts_log_fname, 'w')
        self.ts_log.write('n_buf, buf.pts\n')
        self.n_frames = 0
        self.pipeline = Gst.Pipeline()
        
        def on_new_sample(appsink):
            """
            Function called from the pipeline by appsink.
            Writes the timestampes of frame capture to a log file.
            """
            # Get the buffer
            smp = appsink.emit('pull-sample')
            buf = smp.get_buffer()
            self.n_frames += 1
            self.ts_log.write('%d,%0.9f\n' % 
                              (self.n_frames, np.float64(1e-9) * buf.pts))
            return Gst.FlowReturn.OK

        # Video source:
        self.v4l2src = Gst.ElementFactory.make('v4l2src', None)
        self.v4l2src.set_property('device', video_dev)
        self.v4l2src.set_property('do-timestamp', 'true')
        # Formats available from C920 camera: 
        #   'image/jpeg', 'video/x-h264', 'video/x-raw'
        vid_caps = Gst.Caps.from_string('video/x-h264,'
                                        'width=%d,height=%d,'
                                        'framerate=%d/1' % (640, 480, fps))
        self.vid_filter = Gst.ElementFactory.make('capsfilter', None)
        self.vid_filter.set_property('caps', vid_caps)
        self.vid_parse = Gst.ElementFactory.make('h264parse', None)
        self.ts_sink = Gst.ElementFactory.make('appsink', None)
        # Setting properties of appsink
        # Tell sink to emit signals
        self.ts_sink.set_property('emit-signals', True)
        self.ts_sink.set_property('sync', False)  # No sync
        # Connect appsink to my function (writing timestamps)
        self.ts_sink.connect('new-sample', on_new_sample)

        # Add elements to the pipeline
        self.pipeline.add(self.v4l2src)
        self.pipeline.add(self.vid_filter)
        self.pipeline.add(self.vid_parse)
        self.pipeline.add(self.ts_sink)

        # link
        self.v4l2src.link(self.vid_filter)
        self.vid_filter.link(self.vid_parse)
        self.vid_parse.link(self.ts_sink)
                      
    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def quit(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.ts_log.close()

       
class Webcam_ts_minimal_raw:
    def __init__(self, video_dev='/dev/video0', fps=15):
        """
        """
        ts_log_fname = 'ts_test.log'
                 
        self.ts_log = open(ts_log_fname, 'w')
        self.ts_log.write('n_buf, buf.pts, time.perf_counter()\n')
        self.n_frames = 0
        self.t0 = 0.0
        self.pipeline = Gst.Pipeline()
        
        def on_new_sample(appsink):
            """
            Function called from the pipeline by appsink.
            Writes the timestampes of frame capture to a log file.
            """
            # Get the buffer
            smp = appsink.emit('pull-sample')
            buf = smp.get_buffer()
            pts = np.float64(1e-9) * buf.pts
            t = time.perf_counter() - self.t0
            self.n_frames += 1
            self.ts_log.write('%d,%0.9f,%0.9f\n' % (self.n_frames, pts, t))
            return Gst.FlowReturn.OK

        # Video source:
        self.v4l2src = Gst.ElementFactory.make('v4l2src', None)
        self.v4l2src.set_property('device', video_dev)
        self.v4l2src.set_property('do-timestamp', 'true')
        # Formats available from C920 camera: 
        #   'image/jpeg', 'video/x-h264', 'video/x-raw'
        vid_caps = Gst.Caps.from_string('video/x-raw,'
                                        'format=YUY2,'
                                        'width=%d,height=%d,'
                                        'framerate=%d/1' % (640, 480, fps))
        self.vid_filter = Gst.ElementFactory.make('capsfilter', None)
        self.vid_filter.set_property('caps', vid_caps)
        #self.vid_parse = Gst.ElementFactory.make('jpegparse', None)
        self.ts_sink = Gst.ElementFactory.make('appsink', None)
        # Setting properties of appsink
        # Tell sink to emit signals
        self.ts_sink.set_property('emit-signals', True)
        self.ts_sink.set_property('sync', False)  # No sync
        # Connect appsink to my function (writing timestamps)
        self.ts_sink.connect('new-sample', on_new_sample)

        # Add elements to the pipeline
        self.pipeline.add(self.v4l2src)
        self.pipeline.add(self.vid_filter)
        #self.pipeline.add(self.vid_parse)
        self.pipeline.add(self.ts_sink)

        # link
        self.v4l2src.link(self.vid_filter)
        self.vid_filter.link(self.ts_sink)
        #self.vid_filter.link(self.vid_parse)
        #self.vid_parse.link(self.ts_sink)
                      
    def run(self):
        self.t0 = time.perf_counter()
        self.pipeline.set_state(Gst.State.PLAYING)

    def quit(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.ts_log.close()        

        
def get_video_frames_in_interval(vid_fn, ts_fn, interval=[0, -1]):
    """
    interval -- list or tuple with t0 and t1, the first and last time in the
                interval to extract.
    """

    dtypes = [('frame_n', int), ('ts', np.float64), ('run_ts', np.float64), ('py_ts', np.float64)]
    n_and_timestamps = np.genfromtxt(ts_fn,
                                     dtype=dtypes,
                                     skip_header=2,
                                     delimiter=',')

    reader = imageio.get_reader(vid_fn)
    #import pdb;pdb.set_trace()
    if interval[0] == 0 and interval[1] == -1:
        ix0 = 0
        ix1 = len(n_and_timestamps)-1
    else:
        ix0 = np.flatnonzero(n_and_timestamps['ts'] >= interval[0])[0]
        ix1 = np.flatnonzero(n_and_timestamps['ts'] <= interval[1])[-1]
    frames = []
    #n_and_ts = np.recarray(shape=(), dtype=dtypes)
    for ix in range(ix0, ix1+1):
        frames.append(reader.get_data(ix))

    reader.close()
    n_and_ts = n_and_timestamps[ix0:ix1+1]

    return frames, n_and_ts


class VideoReader:
    """
    http://www.eurion.net/python-snippets/snippet/Seeking%20with%20gstreamer.html
    
    """
    def __init__(self, filename):
        """
        """
        self.readfd, self.writefd = os.pipe() 
        self.frame_size = (480,640,3)
        self.frame_size_bytes = np.prod(self.frame_size)
        self.filesrc = Gst.ElementFactory.make('filesrc', None)
        self.filesrc.set_property('location', filename)
        self.filesrc.set_property('num_buffers', 1)
        self.demuxer = Gst.ElementFactory.make('matroskademux', None)
        self.demuxer.connect('pad-added', self.on_new_pad)
        demux_caps = Gst.Caps.from_string('video/x-theora')
        self.demux_filter = Gst.ElementFactory.make('capsfilter', None)
        self.demux_filter.set_property('caps', demux_caps)
        self.parser = Gst.ElementFactory.make('theoraparse')
        self.decoder = Gst.ElementFactory.make('theoradec', None)
        self.videoconverter = Gst.ElementFactory.make('videoconvert', None)
        self.fdsink = Gst.ElementFactory.make('fdsink', None)
        videoconv_caps = Gst.Caps.from_string('video/x-raw,format=RGB')
        self.fdsink.set_property("fd", self.writefd)
                
        # Create GStreamer pipline
        self.pipeline = Gst.Pipeline()
                     
        self.pipeline.add(self.filesrc)
        self.pipeline.add(self.demuxer)
        self.pipeline.add(self.demux_filter)
        self.pipeline.add(self.parser)
        self.pipeline.add(self.decoder)
        self.pipeline.add(self.videoconverter)
        self.pipeline.add(self.fdsink)
        # link
        self.filesrc.link(self.demuxer)
        self.demux_filter.link(self.parser)
        self.parser.link(self.decoder)
        self.decoder.link(self.videoconverter)
        self.videoconverter.link_filtered(self.fdsink, videoconv_caps)
    
        # Create bus to get events from GStreamer pipeline      
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        #self.bus.connect('message::eos', self.on_finish)
        #self.bus.connect('message::error', self.on_error) 
      
        self.pipeline.set_state(Gst.State.PAUSED)
            
    def get_frame_by_number(self, frame_number):
        self.pipeline.set_state(Gst.State.READY)
        t = self.filesrc.seek_simple(Gst.Format.BUFFERS,
                                      Gst.SeekFlags.FLUSH,
                                      frame_number)
        print('\nt = ',t,'\n')
        self.pipeline.set_state(Gst.State.PLAYING)                                  
        frame = np.fromstring(os.read(self.readfd,
                                      self.frame_size_bytes),
                              dtype='uint8').reshape(self.frame_size)
        self.pipeline.set_state(Gst.State.PAUSED)
        return frame
        
    def get_frame_by_time(self, frame_time):
        """
        frame_time in seconds
        """
        self.pipeline.set_state(Gst.State.READY)        
        t = self.pipeline.seek_simple(Gst.Format.TIME,
                                      Gst.SeekFlags.FLUSH,
                                      frame_time*Gst.SECOND)
        print('\nt = ',t,'\n')
        self.pipeline.set_state(Gst.State.PLAYING)                                  
        frame = np.fromstring(os.read(self.readfd,
                                      self.frame_size_bytes),
                              dtype='uint8').reshape(self.frame_size)
        self.pipeline.set_state(Gst.State.PAUSED)        
        return frame        
    
    def quit(self):
        self.pipeline.set_state(Gst.State.NULL)

    def on_new_pad(self, matroskademux, pad):
        self.demuxer.link(self.demux_filter)
        
    ######################################################
    # Generates the dot file, checks that graphviz in installed
    # and generates a png file, which it then displays the pipeline
    ######################################################
    def on_debug_activate(self):
        fn = 'pipeline-debug-graph'
        fn_dot = "%s/%s.dot" % (os.environ.get("GST_DEBUG_DUMP_DOT_DIR"), fn)
        fn_png = "%s/%s.png" % (os.environ.get("GST_DEBUG_DUMP_DOT_DIR"), fn)
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, fn)
        try:
            os.system("dot -Tpng %s > %s" % (fn_dot, fn_png))
            print('Pipline graph written to %s' % fn_png)
            Gtk.show_uri(None, "file://%s" % fn_png, 0)
        except:
            print('Failure!')
            # check if graphviz is installed with a simple test
            if os.system('which dot'):
                print("Graphviz does not seem to be installed.")

