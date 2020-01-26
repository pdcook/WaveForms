import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import threading

"""
    TODO

    [X] - Re-do everything in terms of single wavelengths, make waveforms exact, add option for special function to playWave (for noise and pulse)
    [ ] - Allow animations for arbitrary waveforms in order for Pulse and WhiteNoise to work
    [ ] - Add LFO frequency bars for Pulse duty cycle (add ability for arbitrary extra parameter vertical sliders (default min/max to 0 to 1 and let the user deal with it)
    [ ] - Add reset button functionality
    [X] - Add frequency slider functionality
    [X] - Add global volume/duration/wavelength view sliders
    [X] - Clean up global sample_rate variable, allow user to set it in the code
    [ ] - Add rectification button on each WaveForm
    [ ] - Add mixer
    [ ] - Fourier transforms

"""

LEFT = 0.05
RIGHT = 0.95
TOP = 0.995
BOTTOM = 0.005

OSC1freq = 440
OSC2freq = 440

DURATION = 10
VOLUME = 0.5
WAVELENGTHS = 5
WAVELENGTHS_MIN = 0
WAVELENGTHS_MAX = 10

SAMPLE_RATE = 44100

COLS = 3

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

        Methods

        NOTES
            - At least one of Teq, Feq, Weq, or waveform must be specified.
            - Teq must be a function of n (index in the series), time, and
                frequency in the form Teq(n,t,f)
            - Feq must be a function of n (index in the series) and frequency
                in the form Feq(n,f)
            - Weq must be a function of t (time) and f (frequency) and in the
                form Weq(t,f,s)
            - Extra arguments (such as pulse widths) required by either Teq or
                Feq may be specified by **kwargs in both the equation's
                function and the WaveForm initialization. Example included.

    """

    def __init__(self, name, Teq = None, Feq = None, Weq = None, \
            waveform = None, ignoreFreq = False, ignoreWavelength = False, \
            animated = False, nmax = 76, **kwargs):
        self.name             = name
        self.Teq              = Teq
        self.Feq              = Feq
        self.Weq              = Weq
        self.nmax             = nmax
        self.waveform         = waveform
        self.ignoreFreq       = ignoreFreq
        self.ignoreWavelength = ignoreWavelength
        self.animated         = animated
        self.kwargs           = kwargs

        if Teq is None and Feq and Weq and waveform is None:
            raise RuntimeError("At least one of Teq, Feq, Weq, or waveform \
must be specified for WaveForm: %s" %name)

        if nmax % 2 != 0:
            raise RuntimeError("nmax must be even for WaveForm: %s" %name)

        # create the animations dict if the plot is to be animated
        if self.animated: self.animations = {}

    def getWaveform(self, volume = 0.5, sample_rate = 44100, \
        duration = None, wavelengths = 1, freq = 440, t = None):
        """
            getWaveform - Returns the waveform in an array

            Parameters
                (Optional) [Default]
                    volume [0.5] - Float between 0.0 and 1.0 (min and max vol)

                    sample_rate [44100] - The sample rate as an integer

                    duration [1/freq] - The duration in seconds

                    freq [440] - Frequency the waveform is played at (in Hz)

                    t [None] - array of times to return the waveform using Weq

            Returns
                waveform - the time-domain waveform as a Numpy array
        """

        # use Weq if t is specified
        if t is not None: return self.Weq(t, freq, **self.kwargs)

        if duration is None: duration = wavelengths/freq
        else: wavelengths = int(np.ceil(duration * freq))

        # if the waveform has been provided, just return it
        if self.waveform is not None and wavelengths is not None and \
            self.ignoreWavelength:

            return self.waveform[:int(SAMPLE_RATE*duration)]

        elif self.waveform is not None: return self.waveform

        # generate the waveform by summing terms in Teq
        if self.Teq is not None:
            waveform = np.sum(self.Teq( \
                np.arange(1,int(self.nmax/2+1))[:,np.newaxis], \
                np.linspace(0, duration, int(round(duration*sample_rate))), \
                freq, **self.kwargs), axis = 0)

        elif self.Feq is not None:
            raise RuntimeError("Generating waveforms from Feq not yet \
supported.")

        elif self.Weq is not None:
            waveform = self.Weq(np.linspace(0, duration, \
                        int(round(duration*sample_rate))), freq, \
                        **self.kwargs)

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

        # opens PlayWave in a separate thread so UI stays responsive
        t = threading.Thread( target = PlayWave, args = \
            [self.getWaveform(volume, sample_rate, duration, None, freq), \
            volume, sample_rate, duration, freq])

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

def Draw(waveforms):
    """
        Draw - Draws the window and runs the main control loop

        Parameters
            None

        Returns
            None
    """

    global OSC1freq
    global OSC2freq

    try:
        ROWS = waveforms.size + 1
    except:
        ROWS = len(waveforms) + 1

    # create figure and axes with ROWS x COLS separate plots
    fig, axes = plt.subplots(ROWS+1, COLS+2)

    # create the mixed scope
    grid = plt.GridSpec(ROWS+1,COLS+2)
    axes[ROWS,0] = plt.subplot(grid[~0,:int(COLS/2)+1])
    axes[ROWS,1] = plt.subplot(grid[~0,~int(COLS/2)-1])

    # grids for the sliders and buttons
    grid2 = fig.add_gridspec(ROWS+1, (COLS+2)*2, left = LEFT, right = RIGHT, \
        bottom = BOTTOM, top = TOP, wspace=1, hspace=1)
    grid3 = fig.add_gridspec((ROWS+1)*4, COLS+2, left = LEFT, \
        right = RIGHT, bottom = BOTTOM, top = TOP, wspace=1, hspace=1)
    grid4 = fig.add_gridspec((ROWS+1)*2, (COLS+2)*2, left = LEFT, \
        right = RIGHT, bottom = BOTTOM, top = TOP, wspace=1, hspace=1)

    # arrays to save the line objects from each oscillator
    OSC1 = np.empty(ROWS, dtype = object)
    OSC2 = np.empty(ROWS, dtype = object)

    # arrays to save the slider and button objects
    sliders      = np.empty(ROWS+1, dtype = object)
    buttons      = np.empty(ROWS+1, dtype = object)
    for i in range(ROWS+1):
        sliders[i] = {}
        buttons[i] = {}

    axes = np.pad(axes,((0,0),(0,COLS+2)), \
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

        for row, button in enumerate(buttons):
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
        OSC1freq = sliders[0]["OSC1"].val
        OSC2freq = sliders[0]["OSC2"].val
        plotWaveforms(False, OSC1freq, OSC2freq)

    def updateVolume(val):
        """
            updateVolume - Updates the volume of playback

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        global VOLUME

        VOLUME = sliders[0]["volume"].val

    def updateDuration(val):
        """
            updateDuration - Updates the duration of playback

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        global DURATION

        DURATION = sliders[0]["duration"].val

    def updateWavelengths(val):
        """
            updateWavelengths - Updates the number of visible wavelengths

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        nonlocal OSC1
        nonlocal OSC2
        global WAVELENGTHS

        WAVELENGTHS = sliders[0]["wavelengths"].val
        plotWaveforms(False, OSC1freq, OSC2freq)

    def updateSampleRate(val):
        """
            updateSampleRate - Updates the sample rate of the waveforms

            Parameters
                val - Value returned by slider which called the function

            Returns
                None
        """

        global SAMPLE_RATE

        SAMPLE_RATE = int(sliders[0]["sample_rate"].val)
        plotWaveforms(False, OSC1freq, OSC2freq)

    def plotWaveforms(init, OSC1freq, OSC2freq):
        """
            plotWaveforms - Draws the waveforms in the window

            Parameters
                init - Boolean which should be True if this is the first plot

                OSC1freq - The frequency of OSC1 in hertz

                OSC2freq - The frequency of OSC2 in hertz

            Returns
                None
        """

        nonlocal OSC1
        nonlocal OSC2

        for row in range(1,ROWS):

            if init:
                OSC1[row], = axes[row, 0].plot(waveforms[row-1].getWaveform(1,\
                                    SAMPLE_RATE, WAVELENGTHS/OSC1freq, \
                                    WAVELENGTHS, OSC1freq))
                OSC2[row], = axes[row, 1].plot(waveforms[row-1].getWaveform(1,\
                                    SAMPLE_RATE, WAVELENGTHS/OSC1freq, \
                                    WAVELENGTHS, OSC2freq))
                # the use of OSC1freq in the calculation for the duration of
                #   OSC2 in the line above may appear to be a mistake, but it
                #   is deliberate in order for the difference in freq. between
                #   OSC1 and OSC2 to be shown

                # set up animated plot if necessary
                if waveforms[row-1].animated:
                    OSC1_meta_func = \
"""
def OSC1_animation_%d(frame, axes, waveforms, OSC1, interval):

    t_size = int(np.round(SAMPLE_RATE*interval))

    t = np.linspace(frame*interval, (frame+1)*interval, t_size)

    waveform_ = waveforms[%d].getWaveform(1,SAMPLE_RATE, WAVELENGTHS/OSC1freq, WAVELENGTHS, OSC1freq, t=t)

    size = int(np.round(SAMPLE_RATE/OSC1freq * WAVELENGTHS))
    xdata = np.arange(waveform_[:size].size)
    OSC1[%d].set_data(xdata, waveform_[:size])
    axes[%d,0].set_xlim([0,waveform_[:size].size-1])
    return OSC1,
"""%(row,row-1,row,row)
                    exec(OSC1_meta_func, globals())

                    interval = (WAVELENGTHS / OSC1freq)*1000    # time per frame in milliseconds
                    exec('waveforms[row-1].animations["OSC1"] = animation.FuncAnimation(fig, OSC1_animation_%d, interval=interval, fargs=(axes, waveforms, OSC1, interval))' %(row))

                    OSC2_meta_func = \
"""
def OSC2_animation_%d(frame, axes, waveforms, OSC2, interval):

    t_size = int(np.round(SAMPLE_RATE*interval))

    t = np.linspace(frame*interval, (frame+1)*interval, t_size)

    waveform_ = waveforms[%d].getWaveform(1,SAMPLE_RATE, WAVELENGTHS/OSC1freq, WAVELENGTHS, OSC2freq, t=t)

    size = int(np.round(SAMPLE_RATE/OSC1freq * WAVELENGTHS))
    xdata = np.arange(waveform_[:size].size)
    OSC2[%d].set_data(xdata, waveform_[:size])
    axes[%d,1].set_xlim([0,waveform_[:size].size-1])
    return OSC2,
"""%(row,row-1,row,row)
                    exec(OSC2_meta_func, globals())

                    interval = (WAVELENGTHS / OSC1freq)*1000    # time per frame in milliseconds
                    exec('waveforms[row-1].animations["OSC2"] = animation.FuncAnimation(fig, OSC2_animation_%d, interval=interval, fargs=(axes, waveforms, OSC2, interval))' %(row))


            else:
                # if the plot is animated it doesn't need to be updated, just
                #    needs to be plotted once
                if waveforms[row-1].animated: continue

                waveform_ = waveforms[row-1].getWaveform(1, \
                                SAMPLE_RATE, WAVELENGTHS/OSC1freq, \
                                WAVELENGTHS, OSC1freq)

                size = int(np.round(SAMPLE_RATE/OSC1freq * WAVELENGTHS))
                xdata = np.arange(waveform_[:size].size)
                OSC1[row].set_data(xdata, waveform_[:size])
                axes[row,0].set_xlim([0,waveform_[:size].size-1])

                waveform_ = waveforms[row-1].getWaveform(1, \
                                SAMPLE_RATE, WAVELENGTHS/OSC1freq, \
                                WAVELENGTHS, OSC2freq)

                size = int(np.round(SAMPLE_RATE/OSC1freq * WAVELENGTHS))
                xdata = np.arange(waveform_[:size].size)
                OSC2[row].set_data(xdata, waveform_[:size])
                axes[row,1].set_xlim([0,waveform_[:size].size-1])

            fig.canvas.draw_idle()

    # plot all waveforms before formatting
    plotWaveforms(True, OSC1freq, OSC2freq)

    # remove ticks from all plots and add sliders
    for row,ax in enumerate(axes):
        for col,a in enumerate(ax):
            if col >= COLS or row == 0:

                try: a.set_visible(False)

                except: pass

                # add oscilator 1 and 2 freq. level
                if row == 0:
                    if 2 < col < 7:
                        axes[row,col] = \
                            fig.add_subplot(grid3[col-3,3:])
                        if col == 3:
                            sliders[row]["volume"] = Slider(axes[row,col], \
                                "Volume", 0, 1, valinit = VOLUME)
                            sliders[row]["volume"].valtext.set_visible(False)
                            sliders[row]["volume"].on_changed(updateVolume)
                        if col == 4:
                            sliders[row]["duration"] = Slider(axes[row,col], \
                                "Duration", 0, 20, valinit = DURATION)
                            sliders[row]["duration"].valtext.set_visible(False)
                            sliders[row]["duration"].on_changed(updateDuration)
                        if col == 5:
                            sliders[row]["wavelengths"] = \
                                Slider(axes[row,col], \
                                "Wavelengths", WAVELENGTHS_MIN, \
                                WAVELENGTHS_MAX, valinit = WAVELENGTHS, \
                                closedmin = False)
                            sliders[row]["wavelengths"].on_changed( \
                                updateWavelengths)
                        if col == 6:
                            sliders[row]["sample_rate"] = \
                                Slider(axes[row,col], "Sample Rate", 20000, \
                                96000, valfmt = "%d Hz", \
                                valinit = SAMPLE_RATE, valstep = 100)
                            sliders[row]["sample_rate"].on_changed( \
                                updateSampleRate)

                    elif col < 2:
                        axes[row,col] = \
                        fig.add_subplot(grid2[row,2*col:2*col+2])


                        if col == 0:
                            sliders[row]["OSC1"] = Slider(axes[row,col], \
                                "OSC1\nFreq.", 16.351597, 4186.009045, \
                                valinit=OSC1freq)
                            sliders[row]["OSC1"].on_changed(updateFreqs)

                        elif col == 1:
                            sliders[row]["OSC2"] = Slider(axes[row,col], \
                                "OSC2\nFreq.", 16.351597, 4186.009045, \
                                valinit=OSC2freq)
                            sliders[row]["OSC2"].on_changed(updateFreqs)

                # add play and reset buttons, and mixing sliders
                else:
                    if col == 2*COLS+2 and row < ROWS:
                        axes[row,col] = fig.add_subplot(grid4[2*row,col:])
                        sliders[row]["OSC1"] = Slider(axes[row,col], \
                            "OSC1\nLevel", 0, 10, valinit=0)

                    elif col == 2*COLS+3 and row < ROWS:
                        axes[row,col] = fig.add_subplot(grid4[2*row+1,col-1:])
                        sliders[row]["OSC2"] = Slider(axes[row,col], \
                            "OSC2\nLevel", 0, 10, valinit=0)

                    elif col == 2*COLS:
                        axes[row,col] = fig.add_subplot(grid2[row,col])
                        buttons[row]["Play"] = Button(axes[row,col], "Play")

                    elif col == 2*COLS+1:
                        axes[row,col] = fig.add_subplot(grid2[row,col])
                        buttons[row]["Reset"] = Button(axes[row,col], "Reset")

                continue

            elif a is None: pass

            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlim(a.get_xlim())
            if row < ROWS:
                if col == 0: a.set_xlim([0,OSC1[row].get_xdata().size-1])
                if col == 1: a.set_xlim([0,OSC2[row].get_xdata().size-1])
            a.set_ylim([-1.2,1.2])
            a.plot([-1*SAMPLE_RATE*100, SAMPLE_RATE*100],[0,0], \
                linestyle = "solid", color = "darkgrey", \
                linewidth=1, zorder=0)

    # assign buttons their functions
    assignButtons()


    # remove space between plots
    plt.subplots_adjust(left = LEFT, right = RIGHT, bottom = BOTTOM, \
                        top = TOP, wspace=0, hspace=0)

    plt.show()

# =========================================================================== #
#                             W A V E F O R M S                               #
# =========================================================================== #

# Triangle WaveForm
def Triangle_Teq(n,t,f):
    return 8*(-(-1)**n)*np.sin((2*n-1)*2*np.pi*f*t)/((2*n-1)*np.pi)**2
def Triangle_Weq(t,f):
    return 2*np.abs(2*(t*f-np.floor(t*f+0.5)))-1

Triangle = WaveForm("Triangle", Weq=Triangle_Weq)

# Pulse WaveForm (example of **kwargs)
def Pulse_Teq(n, t, f, **kwargs):
    d = kwargs["duty"]  # percent duty cycle of the pulse
    return 2*np.sin(n*np.pi*d/100.)* \
        np.cos(n*2*np.pi*f*(t-(d/(100.*f))/2))/(n*np.pi)
def Pulse_Weq(t, f, **kwargs):
    LFOfreq = kwargs["LFOfreq"]
    LFO = Triangle_Weq(t,LFOfreq)
    if np.min(LFO) < 0: LFO -= np.min(LFO)
    LFO /= np.max(np.abs(LFO))

    return ((Saw_Weq(t,f) > LFO).astype(int)).astype(float)

Pulse = WaveForm("Pulse", Weq=Pulse_Weq, LFOfreq = 1, animated = True)

# Square WaveForm
def Square_Teq(n, t, f):
    return 4*np.sin((2*n-1)*2*np.pi*f*t)/((2*n-1)*np.pi)

Square = WaveForm("Square", Square_Teq)

# Sine WaveForm
def Sine_Teq(n, t, f):
    return (n==1).astype(int)*np.sin(2*np.pi*f*t)
def Sine_Weq(t,f):
    return np.sin(2*np.pi*f*t)

Sine = WaveForm("Sine", Weq=Sine_Weq)

# Saw WaveForm
def Saw_Teq(n, t, f):
    return -2*np.sin(n*2*np.pi*f*(t-1/(2*f)))/(n*np.pi)
def Saw_Weq(t, f):
    return -(2/np.pi)*np.arctan((1/np.tan(t*np.pi*f)))

Saw = WaveForm("Saw", Weq=Saw_Weq)

# WhiteNoise WaveForm
WhiteNoise_waveform = np.clip(0.35*np.random.randn(100000),-1,1)
def WhiteNoise_Weq(t, f, **kwargs):
    sigma = kwargs["sigma"] # standard deviation of distribution, uniform if 0
    if sigma <= 0: return np.random.uniform(-1,1,t.size)
    else: return np.clip(sigma*np.random.randn(t.size),-1,1)

WhiteNoise = WaveForm("White Noise", Weq=WhiteNoise_Weq, \
    ignoreWavelength = True, animated = True, sigma = 0.35)


# put WaveForms in order
WAVEFORMS = [Sine, Triangle, Pulse, Saw, WhiteNoise]

# =========================================================================== #

# draw the window
Draw(WAVEFORMS)

# end PyAudio
pa.terminate()
