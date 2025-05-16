import numpy as np

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
            binary_msg = ''.join('0' if phase == 0 else '1' for phase in fm_signal)
            return ' '.join(binary_msg[i:i+8] for i in range(0, len(binary_msg), 8))

        elif fmType == 2:
            binary_msg = ''
            for i in range(0, len(fm_signal), 1):  # Changed to step=1 to match encoding
                phase = fm_signal[i]
                if phase == 0:
                    binary_msg += '00'
                elif phase == 90:
                    binary_msg += '01'
                elif phase == 180:
                    binary_msg += '10'
                elif phase == 270:
                    binary_msg += '11'
                else:
                    raise ValueError(f"Unexpected phase value: {phase}")
            return ' '.join(binary_msg[i:i+8] for i in range(0, len(binary_msg), 8))

        else:
            raise ValueError("Unsupported FM type. Use 1 or 2.")
        
    def pm_to_voltage_array(self, pmSignal, sps, f, fs, A):
        sampleTime = 1/fs
        pmSamples = np.repeat(pmSignal, sps)
        voltageArray = []
        for i in range(0, len(pmSamples)):
            voltageArray.append(A * np.sin(sampleTime * i + np.deg2rad(pmSamples[i])))
        return np.array(voltageArray)
    

    def pm_to_ratios(self, fm_signal):
        return np.array([np.cos(np.radians(phi)) for phi in fm_signal], [np.sin(np.radians(phi)) for phi in fm_signal])
    
    def ratios_to_voltages(self, ratios, A, f, fs, sps):
        timeArray = np.linspace(0, len(ratios)*sps/fs, len(ratios)*sps)
        voltageArray = []
