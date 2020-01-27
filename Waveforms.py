import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import threading

"""
    TODO

    [X] - Re-do everything in terms of single wavelengths, make waveforms exact, add option for special function to playWave (for noise and pulse)
    [X] - Allow animations for arbitrary waveforms in order for Pulse and WhiteNoise to work
    [X] - Add LFO frequency bars for Pulse duty cycle (add ability for arbitrary extra parameter vertical sliders (default min/max to 0 to 1 and let the user deal with it)
    [ ] - Add reset button functionality
    [X] - Add frequency slider functionality
    [X] - Add global volume/duration/wavelength view sliders
    [X] - Clean up global sample_rate variable, allow user to set it in the code
    [ ] - Add rectification button on each WaveForm
    [ ] - Add mixer
    [ ] - Fourier transforms

"""

LEFT   = 0.05   #
RIGHT  = 0.95   #   window
TOP    = 0.995  #   margins
BOTTOM = 0.005  #

OSC1freq = 440  #   starting frequency of OSC1
OSC2freq = 440  #   starting frequency of OSC2

DURATION        = 10    # default duration of playback (arbitrary units)
VOLUME          = 0.5   # default volume (1 is max 0 is min)
WAVELENGTHS     = 5     # default visible wavelength
WAVELENGTHS_MIN = 0     # minimum allowed visible wavelengths
WAVELENGTHS_MAX = 10    # maximum allowed visible wavelengths

SAMPLE_RATE = 44100     # default sample rate in Hz

FIG       = None
AXES      = None
OSC1      = None
OSC2      = None
SLIDERS   = None
BUTTONS   = None
WAVEFORMS = None

COLS = 3    # number of columns in window
ROWS = None

# start PyAudio
pa = pyaudio.PyAudio()

class WaveForm:
    """
        WaveForm - A waveform which can be played, displayed, and modified

        Attributes
            name - Name of the waveform which will be displayed

            (Optional) [Default]
                Teq - The time-domain equation of the nth term of the Fourier
                        series of the waveform

                Feq - The equation of the nth term of the Fourier series of the
                        waveform

                Weq - A function which returns a the waveform as an array

                waveform - The waveform of a single wavelength as a Numpy array

                ignoreFreq - Boolean to determine if the frequency sliders
                                should be ignored (used for white noise)

                ignoreWavelength - Boolean to determine if the wavelength
                                    slider should be ignored (white noise)

                animated [False] - Boolean to determine if the plot is animated

                nmax [76] - highest order term in the Fourier series / the
                                highest harmonic to be used, must be even

                sliders [None] - dictionary of extra parameter sliders in the
                                    form {key: [OSC, MIN, MAX, INIT]}

        Methods

        NOTES
            - At least one of Teq, Feq, Weq, or waveform must be specified.
            - Teq must be a function of n (index in the series), time,
                frequency, and the oscillator in the form Teq(n,t,f,OSC)
            - Feq must be a function of n (index in the series), frequency, and
                the oscillator in the form Feq(n,f,OSC)
            - Weq must be a function of t (time), f (frequency), and the
                oscillator in the form Weq(t,f, OSC)
            - Extra arguments (such as pulse widths) required by either Teq or
                Feq may be specified by **kwargs in both the equation's
                function and the WaveForm initialization. Example included.
            - These extra arguments may be given sliders with the `sliders`
                dict. Each key must match the corresponding key in `kwargs` and
                the associated value must be a list of [OSC, MIN, MAX, INIT]:
                    OSC  - which oscillator this particular slider belongs to
                    MIN  - minimum value for the slider
                    MAX  - maximum value for the slider
                    INIT - initial value for the slider

    """

    def __init__(self, name, Teq = None, Feq = None, Weq = None, \
            waveform = None, ignoreFreq = False, ignoreWavelength = False, \
            animated = False, nmax = 76, sliders = None, **kwargs):
        self.name             = name
        self.Teq              = Teq
        self.Feq              = Feq
        self.Weq              = Weq
        self.nmax             = nmax
        self.waveform         = waveform
        self.ignoreFreq       = ignoreFreq
        self.ignoreWavelength = ignoreWavelength
        self.animated         = animated
        self.sliders          = sliders
        self.OSC1rectify      = False
        self.OSC2rectify      = False
        self.kwargs           = kwargs

        if Teq is None and Feq and Weq and waveform is None:
            raise RuntimeError("At least one of Teq, Feq, Weq, or waveform \
must be specified for WaveForm: %s" %name)

        if nmax % 2 != 0:
            raise RuntimeError("nmax must be even for WaveForm: %s" %name)

        # create the animations dict if the plot is to be animated
        if self.animated: self.animations = {}

    def getWaveform(self, volume = 0.5, sample_rate = 44100, \
        duration = None, wavelengths = 1, freq = 440, t = None, OSC = 1):
        """
            getWaveform - Returns the waveform in an array

            Parameters
                (Optional) [Default]
                    volume [0.5] - Float between 0.0 and 1.0 (min and max vol)

                    sample_rate [44100] - The sample rate as an integer

                    duration [1/freq] - The duration in seconds

                    freq [440] - Frequency the waveform is played at (in Hz)

                    t [None] - array of times to return the waveform using Weq

                    OSC [1] - which oscillator is making the waveform (1 or 2)

            Returns
                waveform - the time-domain waveform as a Numpy array
        """

        # use Weq if t is specified
        if t is not None: waveform = self.Weq(t, freq, OSC, **self.kwargs)

        # make sure both duration and wavelengths are defined
        if duration is None: duration = wavelengths/freq
        else: wavelengths = int(np.ceil(duration * freq))

        # if the waveform has been provided, just return it
        if self.waveform is not None and wavelengths is not None and \
            self.ignoreWavelength:
            waveform = self.waveform[:int(SAMPLE_RATE*duration)]

        elif self.waveform is not None: wavefrom = self.waveform

        # generate the waveform by summing terms in Teq
        if self.Teq is not None:
            waveform = np.sum(self.Teq( \
                np.arange(1,int(self.nmax/2+1))[:,np.newaxis], \
                np.linspace(0, duration, int(round(duration*sample_rate))), \
                freq, OSC, **self.kwargs), axis = 0)

        # generate waveform by inverse Fourier transform
        elif self.Feq is not None:
            raise RuntimeError("Generating waveforms from Feq not yet \
supported.")

        # generate waveform with Weq
        elif self.Weq is not None:
            waveform = self.Weq(np.linspace(0, duration, \
                        int(round(duration*sample_rate))), freq, OSC, \
                        **self.kwargs)

        if OSC == 1 and self.OSC1rectify: waveform = np.abs(waveform)
        if OSC == 2 and self.OSC2rectify: waveform = np.abs(waveform)

        return waveform

    def playWave(self, volume = 0.5, sample_rate = 44100, duration = 20, \
        freq = 440):
        """
            playWave - Play a WaveForm object, wrapper for PlayWave

            Parameters
                (Optional) [Default]
                    volume [0.5] - Float between 0.0 and 1.0 (min and max vol)

                    sample_rate [44100] - The sample rate as an integer

                    duration [20] - The duration in seconds

                    freq [440] - Frequency the waveform is played at (in Hz)

            Returns
                None
        """

        # open PlayWave in a separate thread so UI stays responsive
        t = threading.Thread(target = PlayWave, args = \
            [self.getWaveform(volume, sample_rate, duration, None, freq, \
            OSC=1), volume, sample_rate, duration, freq])

        t.start()

    def playButton(self, event):
        """
            playButton - Translates a button press to playWave

            Parameters
                event - the event that triggered the button via Matplotlib

            Returns
                None
        """

        global OSC1freq

        self.playWave(VOLUME, SAMPLE_RATE, DURATION, OSC1freq)

    def updateOSC1Rectification(self, event):
        """
            updateOSC1Rectification - Updates the rectification of OSC1

            Parameters
                event - the event that triggered the button via Matplotlib

            Returns
                None
        """

        self.OSC1rectify = not self.OSC1rectify

    def updateOSC2Rectification(self, event):
        """
            updateOSC2Rectification - Updates the rectification of OSC1

            Parameters
                event - the event that triggered the button via Matplotlib

            Returns
                None
        """

        self.OSC2rectify = not self.OSC2rectify

def PlayWave(waveform, volume = 0.5, sample_rate = 44100, duration = 20, \
    freq = 440):
    """
        PlayWave - Play a given waveform. The waveform will be repeated or
                    truncated to reach the desired duration

        Parameters
            waveform - A numpy array of the waveform to be played, must be
                able to be converted to float32

            (Optional) [Default]
                volume [0.5] - Float between 0.0 and 1.0 (min and max vol)

                sample_rate [44100] - The sample rate as an integer

                duration [20] - The duration in seconds

                freq [440] - Frequency the waveform is played at (in Hz)

        Returns
            None
    """

    # normalize waveform
    if np.max(np.abs(waveform)) > 0: waveform /= np.max(np.abs(waveform))

    # extend / truncate to approximate duration
    samples = np.size(waveform)

    current_dur = samples / sample_rate

    repeat = int(duration // current_dur)

    if repeat > 0: waveform = np.tile(waveform, repeat)
    else: waveform = waveform[:int(sample_rate//duration)]

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = pa.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    # play. May repeat with different volume values (if done interactively)
    stream.write(volume*waveform.astype(np.float32))

    stream.stop_stream()
    stream.close()

def plotWaveforms(init, fig, axes, OSC1freq, OSC2freq):
    """
        plotWaveforms - Draws the waveforms in the window

        Parameters
            init - Boolean which should be True if this is the first plot

            fig - The figure to plot on

            axes - Axes of the figure

            OSC1freq - The frequency of OSC1 in hertz

            OSC2freq - The frequency of OSC2 in hertz

        Returns
            None
    """

    global OSC1
    global OSC2

    for row in range(1,ROWS):

        if init:
            OSC1[row], = axes[row, 0].plot(WAVEFORMS[row-1].getWaveform(1,\
                                SAMPLE_RATE, WAVELENGTHS/OSC1freq, \
                                WAVELENGTHS, OSC1freq, OSC=1))
            OSC2[row], = axes[row, 1].plot(WAVEFORMS[row-1].getWaveform(1,\
                                SAMPLE_RATE, WAVELENGTHS/OSC1freq, \
                                WAVELENGTHS, OSC2freq, OSC=2))
            # the use of OSC1freq in the calculation for the duration of
            #   OSC2 in the line above may appear to be a mistake, but it
            #   is deliberate in order for the difference in freq. between
            #   OSC1 and OSC2 to be shown

            # set up animated plot if necessary
            #
            # This is a bit convoluted, but each animated subplot requires
            #   a unique animation function. I have decided to use
            #   meta-programming to define these functions. Below, a string
            #   is formatted with values specific to the row/WaveForm. This
            #   string is run as Python code with `exec`, then a secondary
            #   string is run with `exec` which defines and starts the
            #   animation. This method allows for unique functions with
            #   unique names as required. Other than having to use meta-
            #   programming to deal with an arbitrary number / kind of
            #   animated WaveForms, the animation process is pretty
            #   standard for matplotlib.
            if WAVEFORMS[row-1].animated:
                OSC1_meta_func = \
"""
def OSC1_animation_%d(frame, axes, waveforms, OSC1, interval):

    t_size = int(np.round(SAMPLE_RATE*interval))

    t = np.linspace(frame*interval, (frame+1)*interval, t_size)

    waveform_ = waveforms[%d].getWaveform(1,SAMPLE_RATE, WAVELENGTHS/OSC1freq,\
WAVELENGTHS, OSC1freq, t=t, OSC=1)

    size = int(np.round(SAMPLE_RATE/OSC1freq * WAVELENGTHS))
    xdata = np.arange(waveform_[:size].size)
    OSC1[%d].set_data(xdata, waveform_[:size])
    axes[%d,0].set_xlim([0,waveform_[:size].size-1])
    return OSC1,
"""%(row,row-1,row,row)
                exec(OSC1_meta_func, globals())

                # time per frame in milliseconds
                interval = (WAVELENGTHS / OSC1freq)*1000
                exec('WAVEFORMS[row-1].animations["OSC1"] = \
animation.FuncAnimation(fig, OSC1_animation_%d, interval=interval, \
fargs=(axes, WAVEFORMS, OSC1, interval))' %(row))

                OSC2_meta_func = \
"""
def OSC2_animation_%d(frame, axes, waveforms, OSC2, interval):

    t_size = int(np.round(SAMPLE_RATE*interval))

    t = np.linspace(frame*interval, (frame+1)*interval, t_size)

    waveform_ = waveforms[%d].getWaveform(1,SAMPLE_RATE, WAVELENGTHS/OSC1freq,\
WAVELENGTHS, OSC2freq, t=t, OSC=2)

    size = int(np.round(SAMPLE_RATE/OSC1freq * WAVELENGTHS))
    xdata = np.arange(waveform_[:size].size)
    OSC2[%d].set_data(xdata, waveform_[:size])
    axes[%d,1].set_xlim([0,waveform_[:size].size-1])
    return OSC2,
"""%(row,row-1,row,row)
                exec(OSC2_meta_func, globals())

                # time per frame in milliseconds
                interval = (WAVELENGTHS / OSC1freq)*1000
                exec('WAVEFORMS[row-1].animations["OSC2"] = \
animation.FuncAnimation(fig, OSC2_animation_%d, interval=interval, \
fargs=(axes, WAVEFORMS, OSC2, interval))' %(row))

        else:
            # if the plot is animated it doesn't need to be updated, just
            #    needs to be started once
            if WAVEFORMS[row-1].animated: continue

            waveform_ = WAVEFORMS[row-1].getWaveform(1, \
                            SAMPLE_RATE, WAVELENGTHS/OSC1freq, \
                            WAVELENGTHS, OSC1freq, OSC=1)

            size = int(np.round(SAMPLE_RATE/OSC1freq * WAVELENGTHS))
            xdata = np.arange(waveform_[:size].size)
            OSC1[row].set_data(xdata, waveform_[:size])
            axes[row,0].set_xlim([0,waveform_[:size].size-1])

            waveform_ = WAVEFORMS[row-1].getWaveform(1, \
                            SAMPLE_RATE, WAVELENGTHS/OSC1freq, \
                            WAVELENGTHS, OSC2freq, OSC=2)

            size = int(np.round(SAMPLE_RATE/OSC1freq * WAVELENGTHS))
            xdata = np.arange(waveform_[:size].size)
            OSC2[row].set_data(xdata, waveform_[:size])
            axes[row,1].set_xlim([0,waveform_[:size].size-1])

        fig.canvas.draw_idle()

def Draw(waveforms):
    """
        Draw - Draws the window and runs the main control loop

        Parameters
            None

        Returns
            None
    """

    global OSC1
    global OSC2
    global FIG
    global AXES
    global OSC1freq
    global OSC2freq
    global SLIDERS
    global BUTTONS
    global WAVEFORMS
    global ROWS

    WAVEFORMS = waveforms

    try:
        ROWS = waveforms.size + 1
    except:
        ROWS = len(waveforms) + 1

    # create figure and axes with ROWS x COLS separate plots
    FIG, AXES = plt.subplots(ROWS+1, COLS+2)

    # create the mixed scope
    grid = plt.GridSpec(ROWS+1,COLS+2)
    AXES[ROWS,0] = plt.subplot(grid[~0,:int(COLS/2)+1])
    AXES[ROWS,1] = plt.subplot(grid[~0,~int(COLS/2)-1])

    # grids for the sliders and buttons
    grid2 = FIG.add_gridspec(ROWS+1, (COLS+2)*3, left = LEFT, right = RIGHT, \
        bottom = BOTTOM, top = TOP, wspace=1, hspace=2.5)
    grid3 = FIG.add_gridspec((ROWS+1)*4, COLS+2, left = LEFT, \
        right = RIGHT, bottom = BOTTOM, top = TOP, wspace=1, hspace=1)
    grid4 = FIG.add_gridspec((ROWS+1)*2, (COLS+2)*2, left = LEFT, \
        right = RIGHT, bottom = BOTTOM, top = TOP, wspace=1, hspace=1)
    grid5 = FIG.add_gridspec(ROWS+1, (COLS+2)*12, left = LEFT, \
        right = RIGHT, bottom = BOTTOM, top = TOP, wspace=0, hspace=0)
    grid6 = FIG.add_gridspec((ROWS+1)*2, (COLS+2)*3, left = LEFT, \
        right = RIGHT, bottom = BOTTOM, top = TOP, wspace=1, hspace=1)

    # arrays to save the line objects from each oscillator
    OSC1 = np.empty(ROWS, dtype = object)
    OSC2 = np.empty(ROWS, dtype = object)

    # arrays to save the slider and button objects
    SLIDERS      = np.empty(ROWS+1, dtype = object)
    BUTTONS      = np.empty(ROWS+1, dtype = object)
    for i in range(ROWS+1):
        SLIDERS[i] = {}
        BUTTONS[i] = {}

    # add room for the extra objects that require axes (buttons/sliders/etc)
    AXES = np.pad(AXES,((0,0),(0,COLS+2)), \
            mode='constant', constant_values=None)

    def assignButtons():
        """
            assignButtons - Assigns all 'Play' buttons their functions

            Parameters
                volume - The global volume [0.0, 1.0]

                sample_rate - The global sample_rate in Hz

                duration - The duration of played waveforms in seconds

            Returns
                None
        """

        for row, button in enumerate(BUTTONS):
            if button == {}: continue   # skip empty row
            button = button["Play"]     # only modify the 'Play' button

            if row < ROWS:
                button.on_clicked(waveforms[row-1].playButton)

    def updateFreqs(val):
        """
            updateFreqs - Updates the freq. of both OSC1 and OSC2 and plots

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        global OSC1freq
        global OSC2freq

        OSC1freq = SLIDERS[0]["OSC1"].val
        OSC2freq = SLIDERS[0]["OSC2"].val
        plotWaveforms(False, FIG, AXES, OSC1freq, OSC2freq)

    def updateVolume(val):
        """
            updateVolume - Updates the volume of playback

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        global VOLUME

        VOLUME = SLIDERS[0]["volume"].val

    def updateDuration(val):
        """
            updateDuration - Updates the duration of playback

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        global DURATION

        DURATION = SLIDERS[0]["duration"].val

    def updateWavelengths(val):
        """
            updateWavelengths - Updates the number of visible wavelengths

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        global OSC1
        global OSC2
        global WAVELENGTHS

        WAVELENGTHS = SLIDERS[0]["wavelengths"].val
        plotWaveforms(False, FIG, AXES, OSC1freq, OSC2freq)

    def updateSampleRate(val):
        """
            updateSampleRate - Updates the sample rate of the waveforms

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        global SAMPLE_RATE

        SAMPLE_RATE = int(SLIDERS[0]["sample_rate"].val)
        plotWaveforms(False, FIG, AXES, OSC1freq, OSC2freq)


    # plot all waveforms before formatting
    plotWaveforms(True, FIG, AXES, OSC1freq, OSC2freq)

    # remove ticks from all plots and add sliders
    for row,ax in enumerate(AXES):
        for col,a in enumerate(ax):

            # check if the WaveForm on this row needs extra sliders
            if 0 < row < ROWS and col < COLS:
                if waveforms[row-1].sliders is not None:
                    slider_num = 0
                    for param, vals in waveforms[row-1].sliders.items():
                        osc_  = vals[0]
                        min_  = vals[1]
                        max_  = vals[2]
                        init_ = vals[3]

                        if osc_ != col+1: continue

                        slider_num += 1

                        ax_ = FIG.add_subplot(grid5[row,12*(col+1)-slider_num])
                        SLIDERS[row][param] = Slider(ax_, "", min_, max_, \
                                        valinit=init_, orientation='vertical')
                        SLIDERS[row][param].valtext.set_visible(False)

                        meta_func = \
"""
def update_%d(val):

    WAVEFORMS[%d].kwargs["%s"] = SLIDERS[%d]["%s"].val

"""%(row*ROWS+col, row-1, param, row, param)

                        exec(meta_func,globals())
                        exec("SLIDERS[%d]['%s'].on_changed(update_%d)" \
                                %(row,param,row*ROWS+col))

            # check if this is a space where a plot isn't supposed to be
            if col >= COLS or row == 0:

                # some of the AXES don't exist yet, so wrap this in a `try`
                try: a.set_visible(False)
                except: pass

                # add sliders / buttons
                if row == 0:
                    if 2 < col < 7:
                        AXES[row,col] = \
                            FIG.add_subplot(grid3[col-3,3:])

                        if col == 3:
                            SLIDERS[row]["volume"] = Slider(AXES[row,col], \
                                "Volume", 0, 1, valinit = VOLUME)
                            SLIDERS[row]["volume"].valtext.set_visible(False)
                            SLIDERS[row]["volume"].on_changed(updateVolume)

                        if col == 4:
                            SLIDERS[row]["duration"] = Slider(AXES[row,col], \
                                "Duration", 0, 20, valinit = DURATION)
                            SLIDERS[row]["duration"].valtext.set_visible(False)
                            SLIDERS[row]["duration"].on_changed(updateDuration)

                        if col == 5:
                            SLIDERS[row]["wavelengths"] = \
                                Slider(AXES[row,col], \
                                "Wavelengths", WAVELENGTHS_MIN, \
                                WAVELENGTHS_MAX, valinit = WAVELENGTHS, \
                                closedmin = False)
                            SLIDERS[row]["wavelengths"].on_changed( \
                                updateWavelengths)

                        if col == 6:
                            SLIDERS[row]["sample_rate"] = \
                                Slider(AXES[row,col], "Sample Rate", 20000, \
                                96000, valfmt = "%d Hz", \
                                valinit = SAMPLE_RATE, valstep = 100)
                            SLIDERS[row]["sample_rate"].on_changed( \
                                updateSampleRate)

                    elif col < 2:
                        AXES[row,col] = \
                        FIG.add_subplot(grid2[row,3*col:3*col+3])

                        if col == 0:
                            SLIDERS[row]["OSC1"] = Slider(AXES[row,col], \
                                "", 16.351597, 4186.009045, \
                                valinit=OSC1freq)
                            SLIDERS[row]["OSC1"].on_changed(updateFreqs)

                        elif col == 1:
                            SLIDERS[row]["OSC2"] = Slider(AXES[row,col], \
                                "", 16.351597, 4186.009045, \
                                valinit=OSC2freq)
                            SLIDERS[row]["OSC2"].on_changed(updateFreqs)

                # add play and reset buttons, and mixing sliders
                else:
                    if col == 2*COLS+2 and row < ROWS:
                        AXES[row,col] = FIG.add_subplot(grid4[2*row,col:])
                        SLIDERS[row]["OSC1"] = Slider(AXES[row,col], \
                            "OSC1\nLevel", 0, 10, valinit=0)
                        AXES[row,col] = FIG.add_subplot(grid6[2*row,col+3])
                        BUTTONS[row]["OSC1rectify"] = Button(AXES[row,col], \
                            "Rectify\nOSC1")

                        meta_func =\
"""
def update_OSC1_rectification_%d(event):

    WAVEFORMS[%d].OSC1rectify = not WAVEFORMS[%d].OSC1rectify
    plotWaveforms(False, FIG, AXES, OSC1freq, OSC2freq)
"""%(row,row-1,row-1)

                        exec(meta_func, globals())

                        exec("BUTTONS[%d]['OSC1rectify'].on_clicked( \
                                    update_OSC1_rectification_%d)"%(row,row))

                    elif col == 2*COLS+3 and row < ROWS:
                        AXES[row,col] = FIG.add_subplot(grid4[2*row+1,col-1:])
                        SLIDERS[row]["OSC2"] = Slider(AXES[row,col], \
                            "OSC2\nLevel", 0, 10, valinit=0)
                        AXES[row,col] = FIG.add_subplot(grid6[2*row+1,col+2])
                        BUTTONS[row]["OSC2rectify"] = Button(AXES[row,col], \
                            "Rectify\nOSC2")

                        meta_func =\
"""
def update_OSC2_rectification_%d(event):

    WAVEFORMS[%d].OSC2rectify = not WAVEFORMS[%d].OSC2rectify
    plotWaveforms(False, FIG, AXES, OSC1freq, OSC2freq)
"""%(row,row-1,row-1)

                        exec(meta_func, globals())

                        exec("BUTTONS[%d]['OSC2rectify'].on_clicked( \
                                    update_OSC2_rectification_%d)"%(row,row))

                    elif col == 2*COLS:
                        AXES[row,col] = FIG.add_subplot(grid2[row,col+3])
                        BUTTONS[row]["Play"] = Button(AXES[row,col], "Play")

                    elif col == 2*COLS+1:
                        AXES[row,col] = FIG.add_subplot(grid2[row,col+3])
                        BUTTONS[row]["Reset"] = Button(AXES[row,col], "Reset")

                # skip the rest of this loop for a space that isn't a plot
                continue

            elif a is None: pass

            # remove ticks, labels, and set limits
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlim(a.get_xlim())
            if row < ROWS:
                if col == 0: a.set_xlim([0,OSC1[row].get_xdata().size-1])
                if col == 1: a.set_xlim([0,OSC2[row].get_xdata().size-1])
            a.set_ylim([-1.2,1.2])

            # plot gray lines through y = 0
            a.plot([-1*SAMPLE_RATE*100, SAMPLE_RATE*100],[0,0], \
                linestyle = "solid", color = "darkgrey", \
                linewidth=1, zorder=0)

    # assign buttons their functions
    assignButtons()


    # remove space between plots
    plt.subplots_adjust(left = LEFT, right = RIGHT, bottom = BOTTOM, \
                        top = TOP, wspace=0, hspace=0)

    # render (start main UI loop)
    plt.show()

# =========================================================================== #
#                             W A V E F O R M S                               #
# =========================================================================== #

############################# Triangle WaveForm ###############################

def Triangle_Teq(n,t,f,OSC):
    return 8*(-(-1)**n)*np.sin((2*n-1)*2*np.pi*f*t)/((2*n-1)*np.pi)**2
def Triangle_Weq(t,f,OSC):
    return 2*np.abs(2*(t*f-np.floor(t*f+0.5)))-1

Triangle = WaveForm("Triangle", Weq=Triangle_Weq)

###############################################################################

################### Pulse WaveForm (example of **kwargs) ######################

def Pulse_Teq(n, t, f, OSC, **kwargs):
    d = kwargs["duty"]  # percent duty cycle of the pulse
    return 2*np.sin(n*np.pi*d/100.)* \
        np.cos(n*2*np.pi*f*(t-(d/(100.*f))/2))/(n*np.pi)
def Pulse_Weq(t, f, OSC, **kwargs):
    if OSC == 1: LFOfreq = kwargs["LFO1freq"]
    if OSC == 2: LFOfreq = kwargs["LFO2freq"]

    Saw = Saw_Weq(t,f,OSC)
    Saw -= np.min(Saw)
    Saw /= np.max(np.abs(Saw))

    if LFOfreq > 0:
        LFO = Triangle_Weq(t,LFOfreq,OSC)
        if np.min(LFO) < 0: LFO -= np.min(LFO)
        LFO /= np.max(np.abs(LFO))

        return ((Saw > LFO).astype(int)).astype(float)

    # if LFOfreq = 0, just return a 50% pulse (square wave)
    else: return ((Saw > 0.50).astype(int)).astype(float)

sliders = {"LFO1freq": [1,0,10,1], "LFO2freq": [2,0,10,5]}

Pulse = WaveForm("Pulse", Weq=Pulse_Weq, LFO1freq = 1, LFO2freq = 5, \
                    animated = True, sliders = sliders)

###############################################################################

############################## Square WaveForm ################################

def Square_Teq(n, t, f):
    return 4*np.sin((2*n-1)*2*np.pi*f*t)/((2*n-1)*np.pi)

Square = WaveForm("Square", Square_Teq)

###############################################################################

############################### Sine WaveForm #################################

def Sine_Teq(n, t, f, OSC):
    return (n==1).astype(int)*np.sin(2*np.pi*f*t)
def Sine_Weq(t,f,OSC):
    # using -cos to make sure all waveforms are in phase
    return -np.cos(2*np.pi*f*t)

Sine = WaveForm("Sine", Weq=Sine_Weq)

###############################################################################

############################### Saw WaveForm ##################################

def Saw_Teq(n, t, f, OSC):
    return -2*np.sin(n*2*np.pi*f*(t-1/(2*f)))/(n*np.pi)
def Saw_Weq(t, f, OSC):
    return -(2/np.pi)*np.arctan((1/np.tan(t*np.pi*f)))

Saw = WaveForm("Saw", Weq=Saw_Weq)

###############################################################################

################ WhiteNoise WaveForm (example of **kwargs) ####################

WhiteNoise_waveform = np.clip(0.35*np.random.randn(100000),-1,1)
def WhiteNoise_Weq(t, f, OSC, **kwargs):
    # standard deviation of distribution, uniform if 0
    if OSC==1: sigma = kwargs["OSC1sigma"]
    elif OSC==2: sigma = kwargs["OSC2sigma"]
    if sigma <= 0: return np.random.uniform(-1,1,t.size)
    else: return np.clip(sigma*np.random.randn(t.size),-1,1)

sliders = {"OSC1sigma":[1,0,1,0.35], "OSC2sigma":[2,0,1,0.35]}

WhiteNoise = WaveForm("White Noise", Weq=WhiteNoise_Weq, \
    ignoreWavelength = True, animated = True, sliders = sliders, \
    OSC1sigma = 0.35, OSC2sigma = 0.35)

###############################################################################

# put WaveForms in order
WAVEFORMS = [Sine, Triangle, Pulse, Saw, WhiteNoise]

# =========================================================================== #

# draw the window
Draw(WAVEFORMS)

# end PyAudio
pa.terminate()
