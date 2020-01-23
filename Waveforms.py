import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

ROWS = 8
COLS = 2

# start PyAudio
#pa = pyaudio.PyAudio()

class WaveForm:
    """
        WaveForm - A waveform which can be played, displayed, and modified

        Attributes
            name - Name of the waveform which will be displayed

            (Optional)
                Teq - The time-domain equation of the nth term of the Fourier
                        series of the waveform

                Feq - The equation of the nth term of the Fourier series of the
                        waveform

        Methods

        NOTES
            - At least one of Teq and Feq must be specified.
            - Teq must be a function of n (index in the series), time, and
                frequency in the form Teq(n,t,f)
            - Feq must be a function of n (index in the series) and frequency
                in the form Feq(n,f)
            - Extra arguments (such as pulse widths) required by either Teq or
                Feq may be specified by **kwargs in both the equation's
                function and the WaveForm initialization. Example included.

    """

    def __init__(self, name, Teq = None, Feq = None, **kwargs):
        self.name   = name
        self.Teq    = Teq
        self.Feq    = Feq
        self.kwargs = kwargs

        if Teq is None and Feq is None:
            raise RuntimeError("At least one of Teq and Feq must be specified \
for WaveForm: %s" %name)

def PlayWave(waveform, volume = 0.5, sample_rate = 44100, duration = 20, \
    freq = 440):
    """
        PlayWave - Play a given waveform. The waveform will be repeated or
                    truncated to reach the desired duration

        Parameters
            waveform - A numpy array of the waveform to be played, must be able
                to be converted to float32

            (Optional)
                volume - Float between 0.0 and 1.0 (min and max volume)

                sample_rate - The sample rate as an integer

                duration - The duration in seconds

                freq - Frequency the waveform is played at


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
    stream.write(volume*waveform)

    stream.stop_stream()
    stream.close()

def Draw():
    """
        Draw - Draws the window and runs the main control loop

        Parameters
            None

        Returns
            None
    """

    # create figure and axes with ROWS x COLS separate plots
    fig, axes = plt.subplots(ROWS+1, COLS+2)

    # create the mixed scope
    grid = plt.GridSpec(ROWS+1,COLS+2)
    axes[ROWS,0] = plt.subplot(grid[~0,:int(COLS/2)+1])
    axes[ROWS,1] = plt.subplot(grid[~0,~int(COLS/2):])

    grid2 = fig.add_gridspec(ROWS+1, COLS+2, left = 0.125, right = 0.9, \
        bottom = 0.1, top = 0.9, wspace=1, hspace=1)

    # plot all waveforms before formatting
    axes[0,0].plot(np.sin(2*np.pi*np.linspace(0,1,100)))

    sliders = [0 for _ in range(ROWS)]
    buttons = [0 for _ in range(ROWS)]

    # remove ticks from all plots
    for row,ax in enumerate(axes):
        for col,a in enumerate(ax):
            if col >= COLS and row < ROWS:
                a.set_visible(False)
                axes[row,col] = fig.add_subplot(grid2[row,col])
                if col == COLS+1: sliders[row] = Slider(axes[row,col], \
                    "Level", 0, 10, valinit=5)
                if col == COLS: buttons[row] = Button(axes[row,col], "Play")
                continue
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlim(a.get_xlim())
            a.plot([a.get_xlim()[0], a.get_xlim()[~0]],[0,0], \
                linestyle = "solid", color = "darkgrey", linewidth=1, zorder=0)


    # remove space between plots
    plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, \
                            wspace=0, hspace=0)


    plt.show()

# Triangle WaveForm
def Triangle_Teq(n,t,f):
    return 8*(-(-1)**n)*sin((2*n-1)*2*pi*f*x)/((2*n-1)*pi)**2

Triangle = WaveForm("Triangle", Triangle_Teq)

# Pulse WaveForm (example of **kwargs)
def Pulse_Teq(n, t, f, **kwargs):
    d = kwargs["duty"]  # percent duty cycle of the pulse
    return 2*sin(n*pi*d/100.)*cos(n*2*pi*f*(x-(d/(100.*f))/2))/(n*pi)

Pulse = WaveForm("Pulse", Pulse_Teq, duty = 50)

# end PyAudio
#pa.terminate()
