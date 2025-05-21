import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class PMConverter:
    def __init__(self):
        pass

    @staticmethod
    def text_to_binary(text):
        return ' '.join(format(ord(char), '08b') for char in text)

    @staticmethod
    def binary_to_text(binary):
        chars = binary.split()
        return ''.join(chr(int(b, 2)) for b in chars)

    def binary_to_pm(self, binary_msg, fmType):
        if fmType == 1:
            # FM Type 1: BPSK
            binary_msg_nospace = binary_msg.replace(' ', '')
            return np.array([0 if bit == '0' else 180 for bit in binary_msg_nospace])

        elif fmType == 2:
            # FM Type 2: QPSK-like mapping
            binary_msg_nospace = binary_msg.replace(' ', '')
            phases = np.array([
                0 if binary_msg_nospace[i:i+2] == '00' else
                90 if binary_msg_nospace[i:i+2] == '01' else
                180 if binary_msg_nospace[i:i+2] == '10' else
                270
                for i in range(0, len(binary_msg_nospace), 2)
            ])
            return phases

        else:
            raise ValueError("Unsupported FM type. Use 1 or 2.")

    def pm_to_binary(self, fm_signal, fmType):
        if fmType == 1:
            binary_msg = ''.join('0' if np.abs(phase) <= 1 else '1' for phase in fm_signal)
            return ' '.join(binary_msg[i:i+8] for i in range(0, len(binary_msg), 8))

        elif fmType == 2:
            binary_msg = ''
            for i in range(0, len(fm_signal), 1):  # Changed to step=1 to match encoding
                phase = fm_signal[i]
                if phase + 10 <= 0:
                    phase += 360
                if np.abs(phase) <= 1:
                    binary_msg += '00'
                elif np.abs(phase - 90) <= 1:
                    binary_msg += '01'
                elif np.abs(phase - 180) <= 1 or np.abs(phase + 180) <= 1:
                    binary_msg += '10'
                elif np.abs(phase - 270) <= 1:
                    binary_msg += '11'
                else:
                    raise ValueError(f"Unexpected phase value: {phase}")
            return ' '.join(binary_msg[i:i+8] for i in range(0, len(binary_msg), 8))
        elif fmType == 3:
            binary_msg = ''
            for i in range(0, len(fm_signal), 1):  # Changed to step=1 to match encoding
                phase = fm_signal[i]
                if phase == 0 :
                    binary_msg += '000'
                elif phase == 45:
                    binary_msg += '001'
                elif phase == 90 :
                    binary_msg += '010'
                elif phase == 135 :
                    binary_msg += '011'
                elif phase ==180 :
                    binary_msg += '100'
                elif phase ==225 :
                    binary_msg += '101'
                elif phase ==270 :
                    binary_msg += '110'
                elif phase == 315 :
                    binary_msg += '111'
            
                else:
                    raise ValueError(f"Unexpected phase value: {phase}")
            return ' '.join(binary_msg[i:i+8] for i in range(0, len(binary_msg), 8))            
        else:
            raise ValueError("Unsupported FM type. Use 1 or 2.")
        
    def pm_to_voltage_array(self, pmSignal, sps, f, fs, A, pmConfigurationSignal):
        pmSignalFull = np.concatenate([pmConfigurationSignal, pmSignal])
        num_samples = len(pmSignalFull) * sps
        duration = num_samples / fs
        tArray = np.arange(num_samples) / (num_samples / duration)

        voltageArray = []
        phases = []
        for i in range(0, len(tArray)):
            phase = pmSignalFull[i//sps]
            phases.append(phase)
            voltageArray.append(A * np.sin(2 * np.pi * f * tArray[i] + np.deg2rad(phase)))

        plt.scatter(np.arange(len(phases)), phases)
        plt.show()

        return np.array(voltageArray)

    def pm_to_ratios(self, fm_signal):
        return np.array([np.cos(np.radians(phi)) for phi in fm_signal], [np.sin(np.radians(phi)) for phi in fm_signal])
    
    def ratios_to_voltages(self, ratios, A, f, fs, sps):
        timeArray = np.linspace(0, len(ratios)*sps/fs, len(ratios)*sps)
        voltageArray = []

    def lock_in_amplifier(self, voltages, samplerate, carrierFrequency, t):

        # Mix (multiply) input with references
        ref_sin = np.sin(t * 2 * np.pi * carrierFrequency)
        ref_cos = np.cos(t * 2 * np.pi * carrierFrequency)

        X = voltages * ref_cos
        Y = voltages * ref_sin

        time = len(voltages) / samplerate
        X_amp = np.sum(X)#1 / time * np.trapezoid(t, X)
        Y_amp = np.sum(Y)#1 / time * np.trapezoid(t, Y)

        #print(X_amp, Y_amp)
        # Low-pass filter the mixed signals to extract DC component
        def lowpass(signal, cutoff=carrierFrequency / 5, order=3):
            nyquist = 0.5 * samplerate
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low')
            return filtfilt(b, a, signal)

        X_filtered = lowpass(X)
        Y_filtered = lowpass(Y)



        # Compute amplitude and phase
        amplitude = 2 * np.sqrt(np.mean(X_filtered**2 + Y_filtered**2))
        phase = np.arctan2(np.mean(Y_filtered), np.mean(X_filtered))

        amplitude = 2 * np.sqrt(X_amp**2 + Y_amp**2)
        phase = np.arctan2(X_amp, Y_amp)

        return amplitude, phase

    def retrieve_phases(self, data, samplerate, sps, carrierFrequency):
        amplitudes = []
        phases = []
        times = []

        timeArray = np.arange(len(data)) / samplerate

        #print('retrieve duration:', timeArray[-1])

        ref_sin = np.sin(timeArray * 2 * np.pi * carrierFrequency)
        ref_cos = np.cos(timeArray * 2 * np.pi * carrierFrequency)

        n_chunks = len(data) // sps

        #print(n_chunks)
        for i in range(n_chunks):
            start = int((i) * sps)
            end = (i + 1) * sps
            chunk = data[start:end]
            ref_cos_chunk = ref_cos[start:end]
            ref_sin_chunk = ref_sin[start:end]
            #plt.plot(range(start, end), chunk)
            #plt.plot(range(start, end), ref_sin_chunk)
            #plt.plot(range(start, end), ref_cos_chunk)
            #plt.show()

            amp, phase = self.lock_in_amplifier(chunk, samplerate, carrierFrequency, timeArray[start:end])
            #print(phase)

            amplitudes.append(amp)
            phases.append(phase)

            # Time = center of the chunk
            center_time = (start + end) / 2 / samplerate
            times.append(center_time)

        return np.array(times), np.array(amplitudes), np.array(phases)
    
    def configuring_signal(self, samplerate, sps, time):
        timePerSample = 1 / samplerate
        numZeroSymbols = int(time / (timePerSample * sps))
        numZeroSymbols += 8 - (numZeroSymbols % 8) - 2
        signalZeroes = np.full(numZeroSymbols, 0)
        signalEnd = np.full(10, 180)
        signal = np.concatenate([signalZeroes, signalEnd])
        
        return signal
    
    def configure_signal(self, data, samplerate, carrierfrequency, sps, numConfigOnes):
        confStartIndex = int(np.argmax(data > 0.1))
        confEndIndex = int(confStartIndex + 5 * sps)
        tArray = np.arange(len(data)) / samplerate

        lockInAmpData = [self.lock_in_amplifier(
                                            data[i:i+int(0.8*sps)], 
                                            samplerate,
                                            carrierfrequency, 
                                            tArray[i:i+int(0.8*sps)]
                                            ) 
                    for i in range(confStartIndex, confEndIndex)]
        print(lockInAmpData)
        plt.scatter(range(0, len(lockInAmpData[0])), lockInAmpData[0])
        plt.show()