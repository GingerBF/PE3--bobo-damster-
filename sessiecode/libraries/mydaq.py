import matplotlib.pyplot as plt
import numpy as np
import nidaqmx as dx
from scipy.signal import sawtooth, square
import libraries as H
from scipy.fft import rfft, rfftfreq, irfft
from time import sleep
from scipy.io.wavfile import write
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar
import time

class MyDAQ:
    def __init__(self):
        self.__samplerate = None
        self.__name = None

    @property
    def samplerate(self) -> int:
        return self.__samplerate

    @samplerate.setter
    def samplerate(self, newSamplerate: int) -> None:
        assert isinstance(newSamplerate, int), "Samplerate should be an integer."
        assert newSamplerate > 0, "Samplerate should be positive."
        self.__samplerate = newSamplerate

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, newName: str) -> None:
        assert isinstance(newName, str), "Name should be a string."
        self.__name = newName

    def _addOutputChannels(self, task: dx.task.Task, channels: str | list[str]) -> None:
        """
        Add output channels to the DAQ
        """
        assert not (self.name is None), "Name should be set first."

        # Make sure channels can be iterated over
        if isinstance(channels, str):
            channels = [channels]

        # Iterate over all channels and add to task
        for channel in channels:
            if self.name in channel:
                task.ao_channels.add_ao_voltage_chan(channel)
            else:
                task.ao_channels.add_ao_voltage_chan(f"{self.name}/{channel}")

    def _addInputChannels(self, task: dx.task.Task, channels: str | list[str]) -> None:
        """
        Add input channels to the DAQ
        """
        assert not (self.name is None), "Name should be set first."

        # Make sure channels can be iterated over
        if isinstance(channels, str):
            channels = [channels]

        # Iterate over all channels and add to task
        for channel in channels:
            if self.name in channel:
                task.ai_channels.add_ai_voltage_chan(channel)
            else:
                task.ai_channels.add_ai_voltage_chan(f"{self.name}/{channel}")

    def _configureChannelTimings(self, task: dx.task.Task, samples: int) -> None:
        """
        Set the correct timings for task based on number of samples
        """
        assert not (self.samplerate is None), "Samplerate should be set first."

        task.timing.cfg_samp_clk_timing(
            self.samplerate,
            sample_mode=dx.constants.AcquisitionType.FINITE,
            samps_per_chan=samples,
        )

    @staticmethod
    def convertDurationToSamples(samplerate: int, duration: float) -> int:
        samples = duration * samplerate

        # Round down to nearest integer
        return int(samples)

    @staticmethod
    def convertSamplesToDuration(samplerate: int, samples: int) -> float:
        duration = samples / samplerate

        return duration

    def read(self, duration: float, *channels: str, timeout: float = 300) -> np.ndarray:
        """
        Read from user-specified channels for `duration` seconds
        """

        # Convert duration to samples
        samples = MyDAQ.convertDurationToSamples(self.samplerate, duration)

        # Create read task
        with dx.Task("readOnly") as readTask:
            self._addInputChannels(readTask, channels)
            self._configureChannelTimings(readTask, samples)

            # Now read in data. Use WAIT_INFINITELY to assure ample reading time
            data = readTask.read(number_of_samples_per_channel=samples, timeout=timeout)

        return np.asarray(data)

    def write(self, voltages: np.ndarray, *channels: str) -> None:
        """
        Write `voltages` to user-specified channels.
        """
        samples = max(voltages.shape)

        # Create write task
        with dx.Task("writeOnly") as writeTask:
            self._addOutputChannels(writeTask, channels)
            self._configureChannelTimings(writeTask, samples)

            # Now write the data
            writeTask.write(voltages, auto_start=True)

            # Wait for writing to finish
            sleep(samples / self.samplerate + 1 / 1000)
            writeTask.stop()

    def readwrite(
        self,
        voltages: np.ndarray,
        readChannels: str | list[str],
        writeChannels: str | list[str],
        timeout: float = 300,
    ) -> np.ndarray:
        samples = max(voltages.shape)

        with dx.Task("read") as readTask, dx.Task("write") as writeTask:
            self._addOutputChannels(writeTask, writeChannels)
            self._addInputChannels(readTask, readChannels)

            self._configureChannelTimings(writeTask, samples)
            self._configureChannelTimings(readTask, samples)

            # Start writing. Since reading is a blocking function, there
            # is no need to sleep and wait for writing to finish.

            writeTask.write(voltages)
            
            start_time = time.perf_counter()  
            writeTask.start()
            elapsed_time = time.perf_counter() - start_time
            print(f"writeTask.start took {elapsed_time:.6f} seconds.") 
            elapsed_time = 0
            data = readTask.read(number_of_samples_per_channel=(samples - int(elapsed_time * self.samplerate)), timeout=timeout)

            return np.asarray(data)
        
    def readwrite(
        self,
        voltages: np.ndarray,
        readChannels: str | list[str],
        writeChannels: str | list[str],
        timeout: float = 300,
    ) -> np.ndarray:
        samples = max(voltages.shape)

        with dx.Task("read") as readTask, dx.Task("write") as writeTask:
            self._addOutputChannels(writeTask, writeChannels)
            self._addInputChannels(readTask, readChannels)

            self._configureChannelTimings(writeTask, samples)
            self._configureChannelTimings(readTask, samples)

            # Start writing. Since reading is a blocking function, there
            # is no need to sleep and wait for writing to finish.
            writeTask.write(voltages)

            writeTask.start()
            data = readTask.read(number_of_samples_per_channel=samples, timeout=timeout)

            return np.asarray(data)

    @staticmethod
    def generateWaveform(
        function,
        samplerate: int,
        frequency: float,
        amplitude: float = 1,
        phase: float = 0,
        duration: float = 1,
        phaseInDegrees: bool = True,
    ) -> np.ndarray:
        timeArray = MyDAQ.getTimeArray(duration, samplerate)
        if phaseInDegrees:
            phase = np.deg2rad(phase)

        if not callable(function):
            function = MyDAQ.findFunction(function)

        wave = function(timeArray, amplitude, frequency, phase)

        return timeArray, wave

    @staticmethod
    def findFunction(function: str):
        match function:
            case "sine":
                return lambda x, A, f, p: A * np.sin(2 * np.pi * f * x + p)
            case "square":
                return lambda x, A, f, p: A * square(2 * np.pi * f * x + p)
            case "sawtooth":
                return lambda x, A, f, p: A * sawtooth(2 * np.pi * f * x + p)
            case "isawtooth":
                return lambda x, A, f, p: A * sawtooth(2 * np.pi * f * x + p, width=0)
            case "triangle":
                return lambda x, A, f, p: A * sawtooth(2 * np.pi * f * x + p, width=0.5)
            case _:
                raise ValueError(f"{function} is not a recognized wavefront form")

    @staticmethod
    def getTimeArray(duration: float, samplerate: int) -> np.ndarray:
        steps = MyDAQ.convertDurationToSamples(samplerate, duration)
        return np.linspace(1 / samplerate, duration, steps)

    def __str__(self) -> str:
        """
        Only used for pretty printing of class
        E.g. using `print(MyDAQ)` will neatly print the most important
        properties
        """
        title = f"MyDAQ instance"

        return (
            title
            + f"\n{'=' * len(title)}"
            + f"\nBase name: {self.name}"
            + f"\nSample rate: {self.samplerate}"
        )

    @staticmethod
    def remove_magnitude(complex_coefficients: np.ndarray, threshold=0.1) -> np.ndarray:
        """
        Remove the magnitude information from FFT data while keeping phase intact.
        This sets the magnitude of each frequency component to 1.
        """
        # Get the phase of the complex coefficients
        phase = np.angle(complex_coefficients)

        magnitude = np.abs(complex_coefficients)
        # Recreate the complex coefficients with magnitude 1 but the same phase
        magnitude_removed_coefficients = np.exp(1j * phase) * 0.1*np.max(magnitude) # e^(i*phase)

        return magnitude_removed_coefficients
    
            # Apply the threshold for peak detection
        # normalized_magnitude = np.where(magnitude >= threshold * np.max(magnitude), np.max(magnitude), magnitude)

        # # Recombine the magnitude and phase into a complex array
        # normalized_complex_coefficients = normalized_magnitude * np.exp(1j * phase)

        # return normalized_complex_coefficients
    
    @staticmethod
    def remove_phase(complex_coefficients: np.ndarray) -> np.ndarray:
        """
        Remove phase information from the complex FFT coefficients,
        leaving only the magnitude information.

        Parameters:
        complex_coefficients (np.ndarray): Array of complex FFT coefficients.

        Returns:
        np.ndarray: Modified complex array with only magnitude information.
        """
        # Retain the magnitude and set phase to zero
        magnitude_only = np.abs(complex_coefficients) * np.exp(1j * 0)  # Phase set to 0

        return magnitude_only
    
    @staticmethod
    def integral(x,y):
    #integral functions from scipy will be used to integrate over the datapoints
        return integrate.trapezoid(y, x)

    @staticmethod
    def power(freqs, fft, f, samplesize, delta_f=100):
        """
        Get the integration interval, this is a boolean array of frequencies, which
        is true when a frequency is inside the interval and false otherwise. This is used
        to find the frequencies over which we need to integrate.
        """
        if np.isscalar(f):
            f = [f]
        
        f_num = len(f)
        left_margin = 0
        right_margin = 0
        power_list = []

        normalized_fft = fft / samplesize
        normalized_fft[1:] *= 2

        for i, target_freq in enumerate(f):
            if i > 0:
                left_margin = np.exp(0.5 * np.log(f[i] - f[i - 1]))
            else:
                left_margin = delta_f
            if i < f_num - 1:
                right_margin = np.exp(0.5 * np.log(f[i + 1] - f[i]))
            else:
                right_margin = delta_f
            
            interval = (freqs > target_freq - left_margin) & (freqs < target_freq + right_margin) & (freqs > 0)

            power = integrate.trapezoid(np.abs(normalized_fft[interval])**2, freqs[interval])
            power_list.append(power)

        # Ensure the return is always a NumPy array
        return np.sqrt(np.array(power_list))

    @staticmethod
    def generateMultipleWaveforms(function,
        frequencies: np.ndarray,
        amplitude: float = 1,
        phase: float = 0,
        duration: float = 1,
        samplerate: int = 200000,
        phaseInDegrees: bool = True,
    ) -> np.ndarray:
        timeArray = MyDAQ.getTimeArray(duration, samplerate)
        if phaseInDegrees:
            phase = np.deg2rad(phase)

        if not callable(function):
            function = MyDAQ.findFunction(function)
        wave = np.zeros(len(timeArray))
        for f in frequencies:
            wave += function(timeArray, amplitude, f, phase)

        return timeArray, wave
    
    @staticmethod
    def getPhaseShift(freqs, fft_direct, fft_filter, frequencies):
        valid_indices = freqs > 0
        freqs = freqs[valid_indices]
        fft_direct = fft_direct[valid_indices]
        fft_filter = fft_filter[valid_indices]

        frequencies_indices = np.array([np.abs(freqs - target).argmin() for target in frequencies])
        phaseshifts = []
        for f, i in zip(frequencies, frequencies_indices):
            start = max(i - 30, 0)
            stop = min(i + 30, len(freqs) - 1)

            direct_func = interp1d(np.log(freqs[start:stop]), fft_direct[start:stop])
            filter_func = interp1d(np.log(freqs[start:stop]), fft_filter[start:stop])

            direct_interplolated = direct_func(np.log(f))
            filter_interpolated = filter_func(np.log(f))

            phaseshift = np.rad2deg(np.angle(filter_interpolated / direct_interplolated))
            phaseshifts.append(phaseshift)
        
        return phaseshifts

    @staticmethod
    def getMagnitudePhase(
        data: np.ndarray,
        frequencies: np.ndarray | float = 0.0,
        samplerate: int=200000,
        plots: bool=True,
        res_freq: bool=False,
        cutoff_freq: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:

        direct_data = data[0]
        filter_data = data[1]
        samplesize = len(direct_data)

        direct_fft = np.fft.fft(direct_data)
        filter_fft = np.fft.fft(filter_data)
        freqs = np.fft.fftfreq(samplesize, 1/samplerate)

        direct_powers = MyDAQ.power(freqs, direct_fft, frequencies, samplesize, delta_f=20)
        filter_powers = MyDAQ.power(freqs, filter_fft, frequencies, samplesize, delta_f=20)

        H_magnitudes = filter_powers / direct_powers

        frequencies_indices = np.array([np.abs(freqs - target).argmin() for target in frequencies])

        H_phaseshifts = np.rad2deg(np.angle(filter_fft[frequencies_indices] / direct_fft[frequencies_indices]))

        #H_phaseshifts = MyDAQ.getPhaseShift(freqs, direct_fft, filter_fft, frequencies)

        cutoff_index = np.argmin(np.abs(20 * np.log10(H_magnitudes) + 3))
        cutoff_frequency = frequencies[cutoff_index]

        resonant_index = np.argmax(np.abs(20 * np.log10(H_magnitudes)))
        resonant_frequency = frequencies[resonant_index]

        if plots:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), gridspec_kw={'width_ratios': [2, 2, 2]})
                
            ax1.scatter(frequencies, 20 * np.log10(H_magnitudes), label=r'$\frac{A_{out}}{A_{in}}$')
            ax1.set_xscale('log')
            ax1.set_ylabel(r'$20 \log_{10}|H(f)| \, \mathrm{[dB]}$')
            ax1.set_title('Gain of the Black Box')
            if cutoff_freq:
                ax1.axvline(cutoff_frequency, color='red', linestyle='--', label=f'$f_c={round(cutoff_frequency)}$')
            if res_freq:
                ax1.axvline(resonant_frequency, color='orange', linestyle='--', label=f'$f_c={round(resonant_frequency)}$')
            ax1.legend(fontsize=14)
            ax1.grid()

            ax2.scatter(frequencies, H_phaseshifts)
            ax2.set_xscale('log')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('Phase [Degrees]')
            ax2.set_title('Phase-shift of the Black Box')
            if cutoff_freq:
                ax1.axvline(cutoff_frequency, color='red', linestyle='--', label=f'$f_c={round(cutoff_frequency)}$')
            if res_freq:
                ax1.axvline(resonant_frequency, color='orange', linestyle='--', label=f'$f_c={round(resonant_frequency)}$')
            ax2.axhline(-45, color='purple', linestyle='--', label=f'$-45^\\circ $')
            ax2.legend(fontsize=14)
            ax2.grid()

            ax3 = plt.subplot(133, polar=True)  # Create a polar subplot
            ax3.scatter(np.deg2rad(H_phaseshifts), 20 * np.log10(H_magnitudes), label='Data Points', color='red')
            ax3.set_title('Polar Plot')
            ax3.legend()

            plt.suptitle(f'Bode Plots of the data', fontsize=20)
            plt.tight_layout()
            plt.show()

        
        return H_magnitudes, H_phaseshifts
    
    @staticmethod
    def performRFFT(data: np.ndarray, samplerate : int, norm='forward'):

        complex_coefficients = rfft(data, norm=norm)
        frequencies = rfftfreq(len(data), 1 / samplerate)
        if norm =='forward':
            complex_coefficients[1:-1] *= 2  # Correct scaling to show accurate 
        
        if norm =='backward':
            pass
        return frequencies, complex_coefficients
    
    @staticmethod
    def performIRFFT(complex_coefficients, norm='forward'):
        complex_coefficients[1:-1] /= 2    
        original_data = irfft(complex_coefficients, norm=norm)    
        return original_data
    
    