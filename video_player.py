# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:59:40 2016

@author: hjalmar
"""

import matplotlib
matplotlib.use('TkAgg')
import warnings
from matplotlib import pyplot as plt
from matplotlib import patches
from video_tools import VideoReader
import numpy as np
import time
import os
from multiprocessing import Process, Pipe


class DataPump:

    def __init__(self, fname, t0=0.0):
        """
        """
        self.fname = fname
        self._data_end, self._control_end = Pipe()
        self.process = Process(target=self._read_data, args=())
        self.process.start()
        self._control_end.send((True, t0))
        if self._control_end.poll(5.):
            self.dt = self._control_end.recv()
        else:
            warnings.warn("dt could not be retrived from video, waited 5 s",
                          RuntimeWarning)
            self.dt = np.nan

    def _read_data(self):
        """
        """

        vr = VideoReader(self.fname, color=False)
        self._data_end.send(vr.dt_ns * 1e-9)
        running = True

        while running:

            if self._data_end.poll():   # See if msg sent from get_data
                running, next_t = self._data_end.recv()   # get the time of the frame to read (from get_data)
                if running:
                    data = vr.get_frame(next_t)  # Read video frame
                    curr_t = vr.get_current_position(fmt='time')  # get time of frame from video, should be very close to self.data_t
                    self._data_end.send((data, curr_t))  # Send data via the pipe to get_data

        vr.close()

    def get_data(self, next_t):
        """
        Ask for a future frame and returns the previously asked for.
        """
        # Get previous frame and time of frame via the pipe from self._read_data
        data, curr_t = self._control_end.recv()
        # Tell self._read_data to read a new frame at time next_t
        self._control_end.send((True, next_t))

        return data, curr_t

    def close(self):
        """
        """
        self._control_end.send((False, None))
        self._control_end.close()
        self._data_end.close()
        self.process.join()


class VideoPlayer:

    def __init__(self, fname, t0=0.0, fig_h=6., dpi=85):
        """
        """
        if not os.path.isfile(fname):
            raise FileNotFoundError('No such file: %s' % fname)

        plt.ion()
        self.dp = DataPump(fname, t0=t0)
        self.data, self.data_t = self.dp.get_data(t0)
        self.data_dt = self.dp.dt
        self.data_fps = 1 / self.data_dt
        fig_w = fig_h * self.data.shape[1] / self.data.shape[0]
        self.fig = plt.figure(figsize=[fig_w, fig_h], frameon=False, dpi=dpi)
        self._win_title = '%s -- video fps: %1.1f' % (fname.split('/')[-1], self.data_fps)
        self.fig.canvas.set_window_title('%s -- playback not running, press "m" for menu' % self._win_title)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.text = self.ax.text(.025, .025, 't = %1.2fs' % self.data_t, fontsize=12, color='r')
        self.im = self.ax.imshow(self.data, cmap='Greys_r', extent=[0, 1, 0, 1], aspect='auto')
        self.im.figure.canvas.draw()
        #self.bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.cid = self.im.figure.canvas.mpl_connect('key_press_event', self)
        # menu
        #self._menu_bkg = patches.Rectangle((.5, .5), .45, .45, fc='w', alpha=0.75)
        bs = patches.BoxStyle("Round", pad=0.01)
        self._menu_bkg = patches.FancyBboxPatch((.5, .5), .45, .45, boxstyle=bs, fc='w', alpha=0.75)
        menu_text = ('- Left arrow:  decrease speed (increase negative)\n'
                     '- Right arrow: increase speed (decrease negative)\n'
                     '- Space bar:   pause/un-pause\n'
                     '- m:           show/remove this menu\n'
                     '- q:           quit player\n')
        self._menu_text = self.ax.text(.55, .85, menu_text, fontsize=14, color='g', va='top', ha='left')
        self._menu_title = self.ax.text(.725, .9, 'Menu', fontsize=23, fontweight='bold', color='g', va='bottom', ha='center')
        self.ax.add_patch(self._menu_bkg)
        self.quit = False
        self.pause = True # pause
        self.requested_fps = self.data_fps
        self.requested_dt = self.data_dt
        self.no_new_event = True
        self._show_menu = False
        self._menu_showing = False
        plt.ioff()

    def __call__(self, event):
        """
        """
        self.no_new_event = False
        if event.inaxes != self.im.axes:
            return

        if event.key == 'right':  # right arrow
            self.pause = False # un-pause
            self.requested_fps += (self.data_fps * 0.1)  # Speed up
            self.requested_dt = 1. / self.requested_fps
        elif event.key == 'left':  # left arrow
            self.pause = False # un-pause
            self.requested_fps -= (self.data_fps * 0.1) # Slow down
            self.requested_dt = 1. / self.requested_fps
        elif event.key == ' ': # space bar
            if self.pause:
                self.pause = False # un-pause
            else:
                self.pause = True # pause
        elif event.key == 'q':
            self.quit = True
            self.no_new_event = False
        elif event.key == 'm':
            if self._menu_showing:
                self._show_menu = False
                self.pause = False
            else:
                self._show_menu = True
                self.pause = True
        else:
            self.no_new_event = True

        if np.abs(self.requested_fps) < 0.01:
            self.pause = True

    def run(self):

        self.pause = False # un-pause
        dt = np.nan
        data_t1 = self.data_t

        while (not self.data is None) and (not self.quit):

            t0 = time.perf_counter()

            if self.pause:
                self.fig.canvas.set_window_title('%s -- playback paused, press "m" for menu' % self._win_title)

                if self._show_menu and not self._menu_showing:
                    self._menu_bkg.set_visible(True)
                    self._menu_text.set_visible(True)
                    self.ax.draw_artist(self._menu_bkg)
                    self.ax.draw_artist(self._menu_text)
                    self.ax.draw_artist(self._menu_title)
                    self.ax.figure.canvas.blit(self.ax.bbox)
                    self._menu_showing = True
            else:

                # np.sign() takes care of the playback direction (forward or backward)
                data_t1 += self.data_dt * np.sign(self.requested_fps)

                if data_t1 > 0.0: # no frames at negative times
                    self.data, self.data_t = self.dp.get_data(data_t1)
                    self.text.set_text('t = %1.2fs' % self.data_t)
                    fps = self.data_fps * ((data_t1 - self.data_t) / dt)
                    self.im.set_data(self.data)
                    self.fig.canvas.set_window_title('%s -- playback fps: %1.1f (%1.1f), press "m" for menu' %
                        (self._win_title, self.requested_fps, fps))
                    self.ax.draw_artist(self.im)
                    self.ax.draw_artist(self.text)
                    self.ax.figure.canvas.blit(self.ax.bbox)
                    self._menu_showing = False
                    self._show_menu = False
                else:
                    self.pause = True

            self.no_new_event = True
            self.im.figure.canvas.flush_events()

            t1 = t0 + abs(self.requested_dt)
            # check for new events while waiting, so that the player can respond immediately to input events
            while (time.perf_counter() < t1) and self.no_new_event:
                self.im.figure.canvas.flush_events()
                time.sleep(.001) # 1 ms

            dt = time.perf_counter() - t0

        self.dp.close()
        self.im.figure.canvas.mpl_disconnect(self.cid)
        plt.close(self.fig)
